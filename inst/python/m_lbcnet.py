"""Multi-valued-treatment Local Balance Calibration Network.

This module is intentionally separate from the binary LBC-Net runner.  It fits
one shared softmax generalized propensity-score (GPS) network and implements
the treatment-versus-population moments and joint influence-function inference
described by M-LBCNet.
"""

from __future__ import annotations

import importlib.util
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Load the helpers from this package directory under a private module name.  A
# path-based import prevents an installed package or a cached module with the
# same generic name from replacing the local source-of-truth helper file.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HELPER_PATH = os.path.join(_SCRIPT_DIR, "lbc_helpers.py")
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_lbcnet_local_helpers", _HELPER_PATH
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise ImportError("Unable to load the local LBCNet Python helpers.")
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPERS)

vae = _HELPERS.vae
vae_loss = _HELPERS.vae_loss
omega_calculate = _HELPERS.omega_calculate


_M_LSD_MASS_FLOOR = 1e-8
_M_LSD_VARIANCE_FLOOR = 1e-8
_M_LSD_INVALID_VALUE = 1e8


class MLBCNet(nn.Module):
    """Shared neural network with an epsilon-protected softmax output."""

    def __init__(
        self,
        input_dim: int,
        n_treatments: int,
        hidden_dim: int = 100,
        num_layers: int = 2,
        epsilon: float = 0.001,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if n_treatments < 2:
            raise ValueError("n_treatments must be at least 2.")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")
        if not math.isfinite(float(epsilon)) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative.")
        if n_treatments * epsilon >= 1:
            raise ValueError("n_treatments * epsilon must be strictly less than 1.")

        self.input_dim = int(input_dim)
        self.n_treatments = int(n_treatments)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.register_buffer(
            "epsilon", torch.tensor(float(epsilon), dtype=torch.float32)
        )

        self.initial_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.initial_bn = nn.BatchNorm1d(self.hidden_dim)
        self.initial_activation = nn.ReLU()
        self.middle_layers = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(self.num_layers - 1)
            )
        )
        self.final_layer = nn.Linear(self.hidden_dim, self.n_treatments)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_activation(self.initial_bn(self.initial_layer(x)))
        x = self.middle_layers(x)
        logits = self.final_layer(x)
        soft = torch.softmax(logits, dim=1)
        # Componentwise flooring while preserving the probability simplex.
        return self.epsilon + (1.0 - self.n_treatments * self.epsilon) * soft

    def load_vae_encoder_weights(self, vae_model: nn.Module) -> bool:
        """Initialize the first linear layer from a shape-compatible VAE."""
        first_encoder_layer = vae_model.encoder[0]
        if not isinstance(first_encoder_layer, nn.Linear):
            return False
        if first_encoder_layer.weight.shape != self.initial_layer.weight.shape:
            return False
        with torch.no_grad():
            self.initial_layer.weight.copy_(first_encoder_layer.weight)
            self.initial_layer.bias.copy_(first_encoder_layer.bias)
        return True


def _as_tensor_1d(
    value: Sequence[float], name: str, device: torch.device
) -> torch.Tensor:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite numeric vector.")
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _one_hot(
    treatment: torch.Tensor, n_treatments: int, dtype: torch.dtype
) -> torch.Tensor:
    return torch.nn.functional.one_hot(
        treatment.to(dtype=torch.long), num_classes=n_treatments
    ).to(dtype=dtype)


def m_lbcnet_moments(
    propensity_scores: torch.Tensor,
    treatment: torch.Tensor,
    Z: torch.Tensor,
    ck: torch.Tensor,
    h: torch.Tensor,
    kernel_id: int = 0,
    balance_lambda: float = 1.0,
) -> torch.Tensor:
    """Construct per-observation stacked M-LBCNet moments.

    Returns an ``[N, n_treatments * K * (p + 1)]`` tensor.  Within each
    treatment and grid location, the ``p`` treatment-versus-population balance
    components precede the componentwise calibration contribution.
    """
    if propensity_scores.ndim != 2:
        raise ValueError("propensity_scores must have shape [N, n_treatments].")
    if Z.ndim != 2 or treatment.ndim != 1:
        raise ValueError("Z must be two-dimensional and treatment one-dimensional.")
    n_obs, n_treatments = propensity_scores.shape
    if Z.shape[0] != n_obs or treatment.shape[0] != n_obs:
        raise ValueError("propensity_scores, treatment, and Z must have the same N.")
    if ck.ndim != 1 or h.shape != (n_treatments, ck.numel()):
        raise ValueError("h must have shape [n_treatments, K].")
    if not math.isfinite(float(balance_lambda)) or balance_lambda < 0:
        raise ValueError("balance_lambda must be finite and nonnegative.")

    indicators = _one_hot(treatment, n_treatments, propensity_scores.dtype)
    calibration_scale = math.sqrt(float(balance_lambda))
    denom_ck = ck * (1.0 - ck)
    arm_blocks: List[torch.Tensor] = []

    for treatment_index in range(n_treatments):
        p_t = propensity_scores[:, treatment_index]
        r_t = indicators[:, treatment_index]
        kernel_w_t = omega_calculate(
            p_t, ck, h[treatment_index, :], kernel_id
        )  # [N, K]

        # V_i^(t) = {R_i^(t) / pi_i^(t) - 1} Z_i.
        V_t = (r_t / p_t - 1.0).unsqueeze(1) * Z  # [N, p]
        balance = kernel_w_t.unsqueeze(2) * V_t.unsqueeze(1)  # [N,K,p]

        calibration = (
            kernel_w_t * (r_t - p_t).unsqueeze(1) / denom_ck
        )  # [N,K]
        calibration = calibration_scale * calibration

        # Interleave balance and calibration within each local grid location.
        arm_block = torch.cat(
            (balance, calibration.unsqueeze(2)), dim=2
        ).reshape(n_obs, -1)
        arm_blocks.append(arm_block)

    return torch.cat(arm_blocks, dim=1)


def m_lbcnet_loss(
    propensity_scores: torch.Tensor,
    treatment: torch.Tensor,
    Z: torch.Tensor,
    ck: torch.Tensor,
    h: torch.Tensor,
    kernel_id: int = 0,
    balance_lambda: float = 1.0,
    optimizer_scale: bool = True,
) -> torch.Tensor:
    """Identity-weighted stacked-moment objective.

    The mathematical objective is ``Q* = ||mbar||^2 / (L*K)``.  Training uses
    the permitted common positive factor ``N^2`` so its gradient scale remains
    comparable to the existing binary optimizer, which uses summed moments.
    """
    moments = m_lbcnet_moments(
        propensity_scores,
        treatment,
        Z,
        ck,
        h,
        kernel_id=kernel_id,
        balance_lambda=balance_lambda,
    )
    n_obs, _ = moments.shape
    n_treatments = propensity_scores.shape[1]
    n_grid = ck.numel()
    mbar = moments.mean(dim=0)
    q_star = torch.dot(mbar, mbar) / float(n_treatments * n_grid)
    return q_star * (float(n_obs * n_obs) if optimizer_scale else 1.0)


def m_lbcnet_lsd(
    propensity_scores: torch.Tensor,
    treatment: torch.Tensor,
    Z: torch.Tensor,
    ck: torch.Tensor,
    h: torch.Tensor,
    kernel_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Treatment-versus-population local standardized differences.

    ``Z`` contains only genuine covariates; the artificial training intercept is
    deliberately excluded.  Each local difference uses the binary LBC-Net
    pooled weighted-variance and effective-sample-size standardizer.  A local
    neighborhood with mass at or below ``1e-8`` is assigned ``1e8`` rather
    than being reported as balanced.
    """
    n_obs, n_treatments = propensity_scores.shape
    if Z.ndim != 2 or Z.shape[0] != n_obs:
        raise ValueError("Z must have shape [N, p].")
    indicators = _one_hot(treatment, n_treatments, propensity_scores.dtype)
    lsd_blocks: List[torch.Tensor] = []
    summaries: List[torch.Tensor] = []

    for treatment_index in range(n_treatments):
        p_t = propensity_scores[:, treatment_index]
        r_t = indicators[:, treatment_index]
        kernel_w_t = omega_calculate(
            p_t, ck, h[treatment_index, :], kernel_id
        )  # [N,K]
        # Outside a compact kernel's support the mathematical weight is zero,
        # including the extreme-distance case where float32 overflow can form
        # ``Inf * 0`` in the shared Epanechnikov implementation.
        kernel_w_t = torch.where(
            torch.isnan(kernel_w_t), torch.zeros_like(kernel_w_t), kernel_w_t
        )

        treatment_w = kernel_w_t * (r_t / p_t).unsqueeze(1)
        abs_lsd_t = _m_lbcnet_weighted_standardized_difference(
            Z, treatment_w, kernel_w_t
        )
        lsd_blocks.append(abs_lsd_t)
        summaries.append(torch.stack((abs_lsd_t.max(), abs_lsd_t.mean())))

    lsd_values = torch.stack(lsd_blocks, dim=0)  # [L,K,p]
    lsd_by_treatment = torch.stack(summaries, dim=0)  # [L,2]
    return (
        lsd_values.max(),
        lsd_values.mean(),
        lsd_by_treatment,
        lsd_values,
    )


def _m_lbcnet_weighted_standardized_difference(
    Z: torch.Tensor,
    weights_1: torch.Tensor,
    weights_2: torch.Tensor,
) -> torch.Tensor:
    """Return absolute pooled-ESS standardized differences for all K and p."""
    mass_floor = torch.tensor(
        _M_LSD_MASS_FLOOR, dtype=Z.dtype, device=Z.device
    )
    variance_floor = torch.tensor(
        _M_LSD_VARIANCE_FLOOR, dtype=Z.dtype, device=Z.device
    )
    mass_1 = weights_1.sum(dim=0)
    mass_2 = weights_2.sum(dim=0)
    squared_mass_1 = weights_1.square().sum(dim=0)
    squared_mass_2 = weights_2.square().sum(dim=0)
    valid = (
        torch.isfinite(mass_1)
        & torch.isfinite(mass_2)
        & torch.isfinite(squared_mass_1)
        & torch.isfinite(squared_mass_2)
        & (mass_1 > mass_floor)
        & (mass_2 > mass_floor)
        & (squared_mass_1 > 0)
        & (squared_mass_2 > 0)
    )

    safe_mass_1 = torch.where(valid, mass_1, torch.ones_like(mass_1))
    safe_mass_2 = torch.where(valid, mass_2, torch.ones_like(mass_2))
    safe_squared_mass_1 = torch.where(
        valid, squared_mass_1, torch.ones_like(squared_mass_1)
    )
    safe_squared_mass_2 = torch.where(
        valid, squared_mass_2, torch.ones_like(squared_mass_2)
    )

    mean_1 = (weights_1.transpose(0, 1) @ Z) / safe_mass_1.unsqueeze(1)
    mean_2 = (weights_2.transpose(0, 1) @ Z) / safe_mass_2.unsqueeze(1)
    centered_1 = Z.unsqueeze(1) - mean_1.unsqueeze(0)
    centered_2 = Z.unsqueeze(1) - mean_2.unsqueeze(0)
    variance_1 = (
        weights_1.unsqueeze(2) * centered_1.square()
    ).sum(dim=0) / safe_mass_1.unsqueeze(1)
    variance_2 = (
        weights_2.unsqueeze(2) * centered_2.square()
    ).sum(dim=0) / safe_mass_2.unsqueeze(1)
    variance_1 = variance_1.clamp_min(variance_floor)
    variance_2 = variance_2.clamp_min(variance_floor)

    ess_1 = mass_1.square() / safe_squared_mass_1
    ess_2 = mass_2.square() / safe_squared_mass_2
    valid = valid & torch.isfinite(ess_1) & torch.isfinite(ess_2)
    pooled_variance = (
        ess_1.unsqueeze(1) * variance_1
        + ess_2.unsqueeze(1) * variance_2
    ) / (ess_1 + ess_2).unsqueeze(1)
    values = 100.0 * torch.abs(mean_1 - mean_2) / torch.sqrt(pooled_variance)
    values = torch.nan_to_num(
        values,
        nan=_M_LSD_INVALID_VALUE,
        posinf=_M_LSD_INVALID_VALUE,
        neginf=_M_LSD_INVALID_VALUE,
    )
    return torch.where(
        valid.unsqueeze(1),
        values,
        torch.full_like(values, _M_LSD_INVALID_VALUE),
    )


def treatment_hajek_means(
    Y: torch.Tensor,
    treatment: torch.Tensor,
    propensity_scores: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return all treatment-specific Hajek means and denominator means B_t."""
    n_treatments = propensity_scores.shape[1]
    indicators = _one_hot(treatment, n_treatments, propensity_scores.dtype)
    arm_weights = indicators / propensity_scores
    denominators = arm_weights.sum(dim=0)
    if torch.any(denominators <= 0):
        raise ValueError("Every treatment must have a positive Hajek denominator.")
    means = (arm_weights * Y.unsqueeze(1)).sum(dim=0) / denominators
    B = arm_weights.mean(dim=0)
    return means, B


def fixed_gps_influence_functions(
    Y: torch.Tensor,
    treatment: torch.Tensor,
    propensity_scores: torch.Tensor,
    means: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """Fixed-GPS influence functions for all treatment-specific means."""
    n_treatments = propensity_scores.shape[1]
    indicators = _one_hot(treatment, n_treatments, propensity_scores.dtype)
    return (
        indicators
        / (propensity_scores * B.unsqueeze(0))
        * (Y.unsqueeze(1) - means.unsqueeze(0))
    )


def _flatten_grads_with_zeros(
    grads: Sequence[Optional[torch.Tensor]], params: Sequence[torch.Tensor]
) -> torch.Tensor:
    pieces = [
        torch.zeros_like(param).reshape(-1) if grad is None else grad.reshape(-1)
        for grad, param in zip(grads, params)
    ]
    return torch.cat(pieces)


def joint_mean_inference(
    model: MLBCNet,
    treatment: torch.Tensor,
    Y: torch.Tensor,
    Z_processed: torch.Tensor,
    ck: torch.Tensor,
    h: torch.Tensor,
    kernel_id: int = 0,
    balance_lambda: float = 1.0,
    alpha: float = 0.01,
) -> Dict[str, object]:
    """Joint IF inference using one shared nuisance IF and distinct H_t rows."""
    if not math.isfinite(float(alpha)) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative.")

    model.eval()
    params = tuple(param for param in model.parameters() if param.requires_grad)
    n_parameters = sum(param.numel() for param in params)

    with torch.no_grad():
        gps_fixed = model(Z_processed)
        means_fixed, B_fixed = treatment_hajek_means(
            Y, treatment, gps_fixed
        )
        phi_fixed = fixed_gps_influence_functions(
            Y, treatment, gps_fixed, means_fixed, B_fixed
        )
        moments_numeric = m_lbcnet_moments(
            gps_fixed,
            treatment,
            Z_processed,
            ck,
            h,
            kernel_id=kernel_id,
            balance_lambda=balance_lambda,
        )

    # Exact Jacobian G = d mbar(theta) / d theta', using the identical raw
    # moment definition and scaling as the fitted identity-GMM objective.
    gps_graph = model(Z_processed)
    moments_graph = m_lbcnet_moments(
        gps_graph,
        treatment,
        Z_processed,
        ck,
        h,
        kernel_id=kernel_id,
        balance_lambda=balance_lambda,
    )
    mbar_graph = moments_graph.mean(dim=0)
    jacobian_rows: List[torch.Tensor] = []
    for moment_index in range(mbar_graph.numel()):
        grads = torch.autograd.grad(
            mbar_graph[moment_index],
            params,
            retain_graph=True,
            allow_unused=True,
        )
        jacobian_rows.append(_flatten_grads_with_zeros(grads, params).detach())
    G = torch.stack(jacobian_rows, dim=0)  # [q,d_theta]

    # Every arm has its own derivative H_t of the sample Hajek functional,
    # while every row is differentiated through this same network and GPS.
    means_graph, _ = treatment_hajek_means(Y, treatment, gps_graph)
    H_rows: List[torch.Tensor] = []
    for treatment_index in range(gps_graph.shape[1]):
        grads = torch.autograd.grad(
            means_graph[treatment_index],
            params,
            retain_graph=treatment_index < gps_graph.shape[1] - 1,
            allow_unused=True,
        )
        H_rows.append(_flatten_grads_with_zeros(grads, params).detach())
    H = torch.stack(H_rows, dim=0)  # [L,d_theta]
    H_row_norms = torch.linalg.vector_norm(H, dim=1)
    H_pairwise_distances = torch.pdist(H, p=2)
    minimum_H_row_distance = (
        float(H_pairwise_distances.min().detach().cpu())
        if H_pairwise_distances.numel() > 0
        else float("nan")
    )

    # Existing LBC-Net stabilization philosophy: truncated SVD plus an
    # adaptive ridge in singular space.  Unlike the binary helper, no
    # moment-by-moment rescaling is applied, so fitting and inference retain
    # exactly the same identity-weighted moments.
    U, singular_values, Vh = torch.linalg.svd(G, full_matrices=False)
    if singular_values.numel() == 0 or not torch.isfinite(singular_values).all():
        raise RuntimeError("The M-LBCNet moment Jacobian has an invalid SVD.")
    max_singular = singular_values.max()
    numerical_floor = torch.tensor(
        torch.finfo(singular_values.dtype).eps,
        dtype=singular_values.dtype,
        device=singular_values.device,
    )
    threshold = torch.maximum(1e-3 * max_singular, numerical_floor)
    keep = singular_values >= threshold
    if not bool(torch.any(keep)):
        raise RuntimeError("The M-LBCNet moment Jacobian has numerical rank zero.")

    S_kept = singular_values[keep]
    U_kept = U[:, keep]
    Vh_kept = Vh[keep, :]
    lambda_adaptive = float(alpha) * torch.mean(S_kept.pow(2))

    # H IF_theta without materializing an [N,d_theta] nuisance-IF matrix:
    # -m_i U diag{S/(S^2+lambda)} V' H'.
    h_projection = Vh_kept @ H.transpose(0, 1)  # [rank,L]
    ridge_factor = S_kept / (S_kept.pow(2) + lambda_adaptive)
    chain = -(
        (moments_numeric @ U_kept)
        @ (ridge_factor.unsqueeze(1) * h_projection)
    )  # [N,L]
    influence = phi_fixed + chain

    n_obs = influence.shape[0]
    influence_centered = influence - influence.mean(dim=0, keepdim=True)
    covariance = (
        influence_centered.transpose(0, 1) @ influence_centered
    ) / float(n_obs * n_obs)
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    variances = torch.diagonal(covariance).clamp_min(0.0)
    se_means = torch.sqrt(variances)
    ci_lower = means_fixed - 1.96 * se_means
    ci_upper = means_fixed + 1.96 * se_means

    pairwise: List[Dict[str, float]] = []
    n_treatments = means_fixed.numel()
    for treatment_1 in range(n_treatments):
        for treatment_2 in range(treatment_1 + 1, n_treatments):
            estimate = means_fixed[treatment_1] - means_fixed[treatment_2]
            pair_variance = (
                covariance[treatment_1, treatment_1]
                + covariance[treatment_2, treatment_2]
                - 2.0 * covariance[treatment_1, treatment_2]
            ).clamp_min(0.0)
            pair_se = torch.sqrt(pair_variance)
            pairwise.append(
                {
                    "treatment_1": int(treatment_1),
                    "treatment_2": int(treatment_2),
                    "estimate": float(estimate.detach().cpu()),
                    "se": float(pair_se.detach().cpu()),
                    "ci_lower": float((estimate - 1.96 * pair_se).detach().cpu()),
                    "ci_upper": float((estimate + 1.96 * pair_se).detach().cpu()),
                }
            )

    positive_singular = S_kept[S_kept > 0]
    condition_number = (
        float((positive_singular.max() / positive_singular.min()).detach().cpu())
        if positive_singular.numel() > 0
        else float("inf")
    )
    return {
        "means": means_fixed.detach().cpu().numpy(),
        "se_means": se_means.detach().cpu().numpy(),
        "ci_lower_means": ci_lower.detach().cpu().numpy(),
        "ci_upper_means": ci_upper.detach().cpu().numpy(),
        "covariance_means": covariance.detach().cpu().numpy(),
        "pairwise_ate": pairwise,
        "influence_functions": influence.detach().cpu().numpy(),
        "inference_diagnostics": {
            "n_parameters": int(n_parameters),
            "moment_dimension": int(moments_numeric.shape[1]),
            "jacobian_rank": int(keep.sum().detach().cpu()),
            "largest_singular_value": float(max_singular.detach().cpu()),
            "smallest_retained_singular_value": float(
                S_kept.min().detach().cpu()
            ),
            "retained_condition_number": condition_number,
            "adaptive_ridge": float(lambda_adaptive.detach().cpu()),
            "H_shape": [int(H.shape[0]), int(H.shape[1])],
            "H_row_norms": [
                float(value) for value in H_row_norms.detach().cpu()
            ],
            "minimum_H_row_distance": minimum_H_row_distance,
            "G_shape": [int(G.shape[0]), int(G.shape[1])],
        },
    }


def _validate_runner_inputs(
    Z_array: np.ndarray,
    treatment_array: np.ndarray,
    Y_array: Optional[np.ndarray],
    ck_array: np.ndarray,
    h_array: np.ndarray,
    n_treatments: int,
    epsilon: float,
    balance_lambda: float,
    alpha: float,
) -> None:
    if Z_array.ndim != 2 or Z_array.shape[0] < 2 or Z_array.shape[1] < 1:
        raise ValueError("Z must be a numeric matrix with at least two rows and one column.")
    if not np.isfinite(Z_array).all():
        raise ValueError("Z must contain only finite values.")
    if treatment_array.ndim != 1 or treatment_array.shape[0] != Z_array.shape[0]:
        raise ValueError("treatment must be a length-N vector.")
    if not np.isfinite(treatment_array).all():
        raise ValueError("treatment must contain only finite codes.")
    if not np.equal(treatment_array, np.floor(treatment_array)).all():
        raise ValueError("treatment codes must be integers.")
    if n_treatments < 2:
        raise ValueError("At least two observed treatments are required.")
    codes = treatment_array.astype(np.int64)
    if codes.min() < 0 or codes.max() >= n_treatments:
        raise ValueError("treatment codes must be in 0,...,n_treatments-1.")
    if np.unique(codes).size != n_treatments:
        raise ValueError("Every treatment code must be observed.")
    if Y_array is not None:
        if Y_array.ndim != 1 or Y_array.shape[0] != Z_array.shape[0]:
            raise ValueError("Y must be a length-N numeric vector.")
        if not np.isfinite(Y_array).all():
            raise ValueError("Y must contain only finite values.")
    if ck_array.ndim != 1 or ck_array.size == 0:
        raise ValueError("ck must be a nonempty vector.")
    if not np.isfinite(ck_array).all() or np.any((ck_array <= 0) | (ck_array >= 1)):
        raise ValueError("ck values must be finite and strictly between 0 and 1.")
    if h_array.shape != (n_treatments, ck_array.size):
        raise ValueError("h must have shape [n_treatments, K].")
    if not np.isfinite(h_array).all() or np.any(h_array <= 0):
        raise ValueError("h values must be finite and strictly positive.")
    if not math.isfinite(float(epsilon)) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative.")
    if n_treatments * epsilon >= 1:
        raise ValueError("n_treatments * epsilon must be strictly less than 1.")
    if not math.isfinite(float(balance_lambda)) or balance_lambda < 0:
        raise ValueError("balance_lambda must be finite and nonnegative.")
    if not math.isfinite(float(alpha)) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative.")


def run_m_lbcnet(
    data_df,
    Z_columns: Sequence[str],
    T_column: str,
    Y_column: Optional[str],
    n_treatments: int,
    ck: Sequence[float],
    h: Sequence[Sequence[float]],
    kernel: str = "gaussian",
    gpu: int = 0,
    seed: int = 100,
    hidden_dim: int = 100,
    num_layers: int = 2,
    vae_epochs: int = 250,
    vae_lr: float = 0.01,
    max_epochs: int = 5000,
    lr: float = 0.05,
    weight_decay: float = 1e-5,
    balance_lambda: float = 1.0,
    epsilon: float = 0.001,
    lsd_threshold: float = 2.0,
    alpha: float = 0.01,
    rolling_window: int = 5,
    show_progress: bool = True,
    compute_variance: bool = True,
) -> Dict[str, object]:
    """Fit one joint M-LBCNet GPS and, when requested, joint ATE inference."""
    kernel_ids = {"gaussian": 0, "uniform": 1, "epanechnikov": 2}
    if kernel not in kernel_ids:
        raise ValueError(
            "kernel must be one of 'gaussian', 'uniform', or 'epanechnikov'."
        )
    kernel_id = kernel_ids[kernel]

    n_treatments = int(n_treatments)
    hidden_dim = int(hidden_dim)
    num_layers = int(num_layers)
    vae_epochs = int(vae_epochs)
    max_epochs = int(max_epochs)
    rolling_window = int(rolling_window)
    if hidden_dim < 1 or num_layers < 1:
        raise ValueError("hidden_dim and num_layers must be positive integers.")
    if vae_epochs < 0 or max_epochs < 1:
        raise ValueError("vae_epochs must be nonnegative and max_epochs must be positive.")
    if rolling_window < 1:
        raise ValueError("rolling_window must be positive.")
    numeric_tuning = {
        "vae_lr": vae_lr,
        "lr": lr,
        "weight_decay": weight_decay,
        "lsd_threshold": lsd_threshold,
    }
    if any(not math.isfinite(float(value)) for value in numeric_tuning.values()):
        raise ValueError("Numeric tuning parameters must be finite.")
    if vae_lr <= 0 or lr <= 0 or weight_decay < 0 or lsd_threshold < 0:
        raise ValueError(
            "Learning rates must be positive; weight_decay and lsd_threshold must be nonnegative."
        )

    missing_columns = [
        column
        for column in list(Z_columns) + [T_column]
        if column not in data_df.columns
    ]
    if missing_columns:
        raise ValueError("Missing data columns: " + ", ".join(missing_columns))
    if Y_column is not None and Y_column not in data_df.columns:
        raise ValueError(f"Outcome column '{Y_column}' is missing from data_df.")

    Z_array = data_df.loc[:, list(Z_columns)].to_numpy(dtype=np.float32)
    treatment_array = data_df.loc[:, T_column].to_numpy(dtype=np.float64).reshape(-1)
    Y_array = None
    if Y_column is not None:
        Y_array = data_df.loc[:, Y_column].to_numpy(dtype=np.float32).reshape(-1)
    ck_array = np.asarray(ck, dtype=np.float32).reshape(-1)
    h_array = np.asarray(h, dtype=np.float32)

    _validate_runner_inputs(
        Z_array,
        treatment_array,
        Y_array,
        ck_array,
        h_array,
        n_treatments,
        float(epsilon),
        float(balance_lambda),
        float(alpha),
    )

    if torch.cuda.is_available():
        gpu = int(gpu)
        if gpu < 0 or gpu >= torch.cuda.device_count():
            raise ValueError("gpu is outside the available CUDA device range.")
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    Z_raw = torch.as_tensor(Z_array, dtype=torch.float32, device=device)
    treatment = torch.as_tensor(
        treatment_array.astype(np.int64), dtype=torch.long, device=device
    )
    Y = (
        None
        if Y_array is None
        else torch.as_tensor(Y_array, dtype=torch.float32, device=device)
    )
    ck_tensor = torch.as_tensor(ck_array, dtype=torch.float32, device=device)
    h_tensor = torch.as_tensor(h_array, dtype=torch.float32, device=device)

    n_obs, n_covariates = Z_raw.shape
    Z_sd = Z_raw.std(dim=0, unbiased=True)
    safe_sd = torch.where(Z_sd > 1e-8, Z_sd, torch.ones_like(Z_sd))
    Z_normalized = (Z_raw - Z_raw.mean(dim=0)) / safe_sd
    Z_processed = torch.cat(
        (torch.ones((n_obs, 1), dtype=Z_raw.dtype, device=device), Z_normalized),
        dim=1,
    )
    processed_dim = Z_processed.shape[1]

    # VAE pretraining uses a shape-compatible first encoder layer so that the
    # learned first-layer representation can initialize the GPS network.
    vae_model = vae(processed_dim, processed_dim, hidden_dim=hidden_dim).to(device)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=float(vae_lr))
    vae_model.train()
    for _ in range(vae_epochs):
        vae_optimizer.zero_grad()
        reconstruction, mu, logvar = vae_model(Z_processed)
        pretrain_loss = vae_loss(reconstruction, Z_processed, mu, logvar)
        pretrain_loss.backward()
        vae_optimizer.step()

    gps_model = MLBCNet(
        input_dim=processed_dim,
        n_treatments=n_treatments,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        epsilon=float(epsilon),
    ).to(device)
    vae_weights_loaded = False
    if vae_epochs > 0:
        vae_weights_loaded = gps_model.load_vae_encoder_weights(vae_model)
    optimizer = optim.Adam(
        gps_model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )

    lsd_window: List[float] = []
    early_stopping = False
    epochs_run = 0
    if show_progress:
        pbar = tqdm(
            total=max_epochs,
            desc="Training Progress",
            position=0,
            leave=True,
            bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} "
            "[{rate_fmt} {postfix}]",
        )
        start_time = time.time()

    for epoch in range(max_epochs):
        gps_model.train()
        optimizer.zero_grad()
        gps = gps_model(Z_processed)
        loss = m_lbcnet_loss(
            gps,
            treatment,
            Z_processed,
            ck_tensor,
            h_tensor,
            kernel_id=kernel_id,
            balance_lambda=float(balance_lambda),
            optimizer_scale=True,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("M-LBCNet training produced a non-finite loss.")
        loss.backward()
        optimizer.step()
        epochs_run = epoch + 1

        if show_progress:
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            estimated_total_time = avg_time_per_epoch * max_epochs
            remaining_time = max(0, estimated_total_time - elapsed_time)
            pbar.set_postfix(
                {
                    "Remaining Time (s)": f"{remaining_time:.2f}",
                    "Elapsed Time (s)": f"{elapsed_time:.2f}",
                    "Loss": f"{loss.item():.4f}",
                }
            )
            pbar.update(1)

        if epochs_run % 200 == 0:
            gps_model.eval()
            with torch.no_grad():
                check_gps = gps_model(Z_processed)
                check_max_lsd, _, _, _ = m_lbcnet_lsd(
                    check_gps,
                    treatment,
                    Z_raw,
                    ck_tensor,
                    h_tensor,
                    kernel_id=kernel_id,
                )
            lsd_window.append(float(check_max_lsd.detach().cpu()))
            if len(lsd_window) > rolling_window:
                lsd_window.pop(0)
            if (
                len(lsd_window) == rolling_window
                and float(np.mean(lsd_window)) < float(lsd_threshold)
            ):
                print(
                    f"✅ Stopping early at epoch {epoch + 1} "
                    f"(rolling average max LSD < {lsd_threshold}%)"
                )
                early_stopping = True
                break

    if not early_stopping:
        print(
            "⚠️ Stopping criterion not met at max epochs. "
            "Try increasing `max_epochs` or adjusting `lsd_threshold` "
            "for better convergence."
        )

    if show_progress:
        pbar.close()

    gps_model.eval()
    with torch.no_grad():
        final_gps = gps_model(Z_processed)
        row_sum_error = torch.max(torch.abs(final_gps.sum(dim=1) - 1.0))
        if not torch.isfinite(final_gps).all() or float(row_sum_error.cpu()) > 1e-5:
            raise RuntimeError("Final generalized propensity scores do not form a simplex.")
        final_loss = m_lbcnet_loss(
            final_gps,
            treatment,
            Z_processed,
            ck_tensor,
            h_tensor,
            kernel_id=kernel_id,
            balance_lambda=float(balance_lambda),
            optimizer_scale=True,
        )
        final_lsd_max, final_lsd_mean, lsd_by_arm, lsd_values = m_lbcnet_lsd(
            final_gps,
            treatment,
            Z_raw,
            ck_tensor,
            h_tensor,
            kernel_id=kernel_id,
        )
        observed_gps = final_gps[
            torch.arange(n_obs, device=device), treatment
        ]
        weights = 1.0 / observed_gps

    lsd_summaries = [
        {
            "treatment_code": int(index),
            "max_lsd": float(lsd_by_arm[index, 0].detach().cpu()),
            "mean_lsd": float(lsd_by_arm[index, 1].detach().cpu()),
        }
        for index in range(n_treatments)
    ]

    moment_dimension = int(
        n_treatments * ck_tensor.numel() * (processed_dim + 1)
    )
    result: Dict[str, object] = {
        "propensity_scores": final_gps.detach().cpu().numpy(),
        "weights": weights.detach().cpu().numpy(),
        "total_loss": float(final_loss.detach().cpu()),
        "max_lsd": float(final_lsd_max.detach().cpu()),
        "mean_lsd": float(final_lsd_mean.detach().cpu()),
        "lsd_by_treatment": lsd_summaries,
        "lsd_values": lsd_values.detach().cpu().numpy(),
        "epochs_run": int(epochs_run),
        "early_stopping": bool(early_stopping),
        "vae_weights_loaded": bool(vae_weights_loaded),
        "tensor_shapes": {
            "Z": [int(n_obs), int(processed_dim)],
            "Z_raw": [int(n_obs), int(n_covariates)],
            "treatment": [int(n_obs)],
            "R": [int(n_obs), int(n_treatments)],
            "gps": [int(n_obs), int(n_treatments)],
            "kernel_weights": [int(n_obs), int(ck_tensor.numel())],
            "balance_block": [
                int(ck_tensor.numel()),
                int(processed_dim),
            ],
            "calibration_block": [int(ck_tensor.numel())],
            "stacked_moments": [int(n_obs), moment_dimension],
        },
    }

    if Y is not None:
        print(
            "Starting post-processing: computing treatment-specific means, "
            "pairwise ATEs, and variance..."
        )
        with torch.no_grad():
            means_only, _ = treatment_hajek_means(Y, treatment, final_gps)
        if compute_variance:
            inference = joint_mean_inference(
                gps_model,
                treatment,
                Y,
                Z_processed,
                ck_tensor,
                h_tensor,
                kernel_id=kernel_id,
                balance_lambda=float(balance_lambda),
                alpha=float(alpha),
            )
            result.update(inference)
            result["tensor_shapes"].update(
                {
                    "fixed_IF": [int(n_obs), int(n_treatments)],
                    "H": inference["inference_diagnostics"]["H_shape"],
                    "final_IF": [int(n_obs), int(n_treatments)],
                    "covariance": [int(n_treatments), int(n_treatments)],
                }
            )
        else:
            nan_vector = np.full(n_treatments, np.nan, dtype=float)
            nan_covariance = np.full(
                (n_treatments, n_treatments), np.nan, dtype=float
            )
            pairwise = []
            means_numpy = means_only.detach().cpu().numpy()
            for treatment_1 in range(n_treatments):
                for treatment_2 in range(treatment_1 + 1, n_treatments):
                    pairwise.append(
                        {
                            "treatment_1": treatment_1,
                            "treatment_2": treatment_2,
                            "estimate": float(
                                means_numpy[treatment_1] - means_numpy[treatment_2]
                            ),
                            "se": float("nan"),
                            "ci_lower": float("nan"),
                            "ci_upper": float("nan"),
                        }
                    )
            result.update(
                {
                    "means": means_numpy,
                    "se_means": nan_vector.copy(),
                    "ci_lower_means": nan_vector.copy(),
                    "ci_upper_means": nan_vector.copy(),
                    "covariance_means": nan_covariance,
                    "pairwise_ate": pairwise,
                    "influence_functions": None,
                    "inference_diagnostics": None,
                }
            )
            result["tensor_shapes"].update(
                {
                    "fixed_IF": None,
                    "H": None,
                    "final_IF": None,
                    "covariance": [int(n_treatments), int(n_treatments)],
                }
            )

    print("✅ M-LBCNet training completed successfully.")

    return result


__all__ = [
    "MLBCNet",
    "m_lbcnet_moments",
    "m_lbcnet_loss",
    "m_lbcnet_lsd",
    "treatment_hajek_means",
    "fixed_gps_influence_functions",
    "joint_mean_inference",
    "run_m_lbcnet",
]
