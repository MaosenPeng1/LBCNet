import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm  # Progress bar
import time            # Execution timing

try:
    import torch
except ImportError:
    raise ImportError("🚨 Error: 'torch' is not installed. Install it using: pip install torch")

try:
    import numpy as np
except ImportError:
    raise ImportError("🚨 Error: 'numpy' is not installed. Install it using: pip install numpy")

# Directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from lbc_helpers import * 

def run_lbc_net_surv(
    data_df,
    Z_columns,
    T_column,
    time_column,
    delta_column,
    ck,
    h,
    kernel="gaussian",
    gpu=0,
    seed=100,
    hidden_dim=100,
    L=2,
    vae_epochs=250,
    vae_lr=0.01,
    max_epochs=5000,
    lr=0.05,
    weight_decay=1e-5,
    balance_lambda=1.0,
    alpha=0.01,
    epsilon=0.001,
    lsd_threshold=2,
    rolling_window=5,
    show_progress=True,
    t_grid=None,
):
    """
    Run LBC-Net for propensity score estimation and compute survival difference.

    This function:
      1. Trains a VAE on covariates Z,
      2. Trains an LBC-Net propensity model with local balance (using ck, h),
      3. Computes ATE-type IPW weights (ω*(p) = 1),
      4. Applies an IPW-weighted Nelson–Aalen estimator (Deng & Wang, 2025)
         to obtain S1(t), S0(t), and their difference Δ(t),
      5. Computes influence-function-based variance, SE, and Wald CI for Δ(t).

    Parameters
    ----------
    data_df : pandas.DataFrame
        Data containing treatment, time, delta, and covariates.
    Z_columns : list of str
        Covariate column names.
    T_column : str
        Treatment column name (0/1).
    time_column : str
        Event or censoring time column name.
    delta_column : str
        Event indicator column name (1 = event, 0 = censored).
    ck : list or np.ndarray
        Kernel center values in (0, 1) for local balance constraints.
    h : list or np.ndarray
        Bandwidths associated with ck.
    kernel : {"gaussian", "uniform", "epanechnikov"}, default "gaussian"
        Kernel type for LBC-Net loss and LSD calculation.
    gpu : int, default 0
        GPU device index if CUDA is available. Ignored if CUDA not available.

    seed : int, default 100
        Random seed for reproducibility.
    hidden_dim : int, default 100
        Hidden dimension size for LBC-Net.
    L : int, default 2
        Number of hidden layers for LBC-Net.
    vae_epochs : int, default 250
        Number of epochs for VAE pre-training.
    vae_lr : float, default 0.01
        Learning rate for VAE training.
    max_epochs : int, default 5000
        Maximum epochs for LBC-Net training.
    lr : float, default 0.05
        Learning rate for LBC-Net training.
    weight_decay : float, default 1e-5
        L2 regularization for optimizer.
    balance_lambda : float, default 1.0
        Weight on calibration term in LBC-Net loss.
    alpha : float, default 0.01
        Small ridge penalty factor for stabilizing the chain correction.
    epsilon : float, default 0.001
        Propensity score bounding parameter.
    lsd_threshold : float, default 2
        Early-stopping threshold based on max LSD (%).
    rolling_window : int, default 5
        Rolling window length for LSD-based stopping.
    show_progress : bool, default True
        Whether to show a tqdm progress bar.
    t_grid : array-like or None, default None
        Time grid at which to evaluate S1(t), S0(t), Δ(t).
        Must be set by the R wrapper (or derived there).

    Returns
    -------
    dict
        {
          "propensity_scores": list of float,
          "total_loss": float,
          "max_lsd": float,
          "mean_lsd": float,
          "times": list of float,
          "S1": list of float,
          "S0": list of float,
          "surv_diff": list of float,
          "var_diff": list of float,
          "se": list of float,
          "ci_lower": list of float,
          "ci_upper": list of float
        }
    """

    # -----------------------------
    # 1. Device & random seed setup
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu))

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True

    # -----------------------------
    # 2. Prepare ck, h as tensors
    # -----------------------------
    if not isinstance(ck, (list, tuple, np.ndarray)):
        ck = [ck]
    if not isinstance(h, (list, tuple, np.ndarray)):
        h = [h]

    ck = torch.tensor(ck, dtype=torch.float32, device=device)
    h = torch.tensor(h, dtype=torch.float32, device=device)

    # -----------------------------
    # 3. Convert DataFrame to tensors
    # -----------------------------
    Z_numpy = data_df.loc[:, Z_columns].to_numpy(dtype="float32")
    Z = torch.tensor(Z_numpy, dtype=torch.float32, device=device)

    T_numpy = data_df.loc[:, T_column].to_numpy(dtype="float32")
    T = torch.tensor(T_numpy, dtype=torch.float32, device=device)

    time_numpy = data_df.loc[:, time_column].to_numpy(dtype="float32")
    time_t = torch.tensor(time_numpy, dtype=torch.float32, device=device)

    delta_numpy = data_df.loc[:, delta_column].to_numpy(dtype="float32")
    delta_t = torch.tensor(delta_numpy, dtype=torch.float32, device=device)

    n, p = Z.shape

    # Normalize Z and add intercept
    Z_norm = (Z - Z.mean(dim=0)) / Z.std(dim=0)
    Z_norm = torch.cat([torch.ones((n, 1), device=device), Z_norm], dim=1)
    p += 1  # account for intercept

    # Kernel ID for loss/LSD
    kernel_id = {"gaussian": 0, "uniform": 1, "epanechnikov": 2}[kernel]

    # -----------------------------
    # 4. Train VAE
    # -----------------------------
    vae_model = vae(p, p).to(device)
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=vae_lr)

    vae_model.train()
    for epoch in range(int(vae_epochs)):
        vae_optimizer.zero_grad()
        recon_batch, mu, logvar = vae_model(Z_norm)
        loss_vae = vae_loss(recon_batch, Z_norm, mu, logvar)
        loss_vae.backward()
        vae_optimizer.step()

    # -----------------------------
    # 5. Train LBC-Net propensity model
    # -----------------------------
    ps_model = lbc_net(p, hidden_dim, L, epsilon).to(device)
    optimizer = optim.Adam(ps_model.parameters(), lr=lr, weight_decay=weight_decay)
    ps_model.load_vae_encoder_weights(vae_model.encoder.state_dict())

    lsd_window = []
    early_stopping = False

    if show_progress:
        pbar = tqdm(
            total=int(max_epochs),
            desc="Training LBC-Net",
            position=0,
            leave=True,
            bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{rate_fmt} {postfix}]",
        )
        start_time = time.time()

    ate_flag = 1  # Always ATE-type IPW weights for survival differences

    for epoch in range(int(max_epochs)):
        ps_model.train()
        optimizer.zero_grad()

        outputs = ps_model(Z_norm).squeeze()
        loss = lbc_net_loss(
            outputs,
            T,
            Z_norm,
            ck,
            h,
            ate=ate_flag,
            kernel_id=kernel_id,
            balance_lambda=balance_lambda,
        )

        loss.backward()
        optimizer.step()

        # Progress bar
        if show_progress:
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            estimated_total_time = avg_time_per_epoch * max_epochs
            remaining_time = max(0, estimated_total_time - elapsed_time)

            pbar.set_postfix(
                {
                    "Remaining (s)": f"{remaining_time:.2f}",
                    "Elapsed (s)": f"{elapsed_time:.2f}",
                    "Loss": f"{loss.item():.4f}",
                }
            )
            pbar.update(1)

        # LSD-based early stopping every 200 epochs
        if (epoch + 1) % 200 == 0:
            ps_model.eval()
            with torch.no_grad():
                LSD_max, LSD_mean = lsd_cal(
                    ps_model(Z_norm).squeeze(), T, Z, ck, h, kernel_id, ate_flag
                )
                lsd_window.append(LSD_max)

                if len(lsd_window) > rolling_window:
                    lsd_window.pop(0)

                if len(lsd_window) == rolling_window:
                    mean_lsd_window = torch.mean(torch.stack(lsd_window))
                    if mean_lsd_window < lsd_threshold:
                        print(
                            f"✅ Early stopping at epoch {epoch + 1} "
                            f"(rolling average max LSD < {lsd_threshold}%)"
                        )
                        early_stopping = True
                        break

    if show_progress:
        pbar.close()

    if not early_stopping:
        print(
            "⚠️ LSD stopping criterion not met by max_epochs. "
            "Consider increasing `max_epochs` or adjusting `lsd_threshold`."
        )

    # -----------------------------
    # 6. Final propensity scores + LSD
    # -----------------------------
    with torch.no_grad():
        final_outputs = ps_model(Z_norm).squeeze()
        final_LSD_max, final_LSD_mean = lsd_cal(
            final_outputs, T, Z, ck, h, kernel_id, ate_flag
        )
        ps = final_outputs.detach().cpu().numpy()

    # -----------------------------
    # 7. Survival estimands (IPW NA) + IF-based variance
    # -----------------------------
    print("Starting post-processing: computing treatment effect and variance...")

    if t_grid is None:
        raise ValueError("t_grid must be provided by the calling R wrapper.")

    t_grid = np.asarray(t_grid, dtype="float32")
    t_grid_t = torch.tensor(t_grid, dtype=torch.float32, device=device)

    with torch.no_grad():
        # IPW-weighted Nelson–Aalen survival curves and difference
        S1_hat, S0_hat, diff_hat = ipw_surv_na(
            time_t,
            delta_t,
            T,
            final_outputs,
            t_grid_t,
        )

        # Plug-in IF for survival difference treating PS as fixed
        phi_ipw = plugin_if_surv(
            time_t,
            delta_t,
            T,
            final_outputs,
            t_grid_t,
        )

    # IF-based variance accounting for LBC-Net PS estimation
    var_diff_t = if_var_surv(
        model=ps_model,
        A=T,
        time=time_t,
        delta=delta_t,
        Z=Z_norm,
        ck=ck,
        h=h,
        phi_ipw=phi_ipw,
        t_grid = t_grid_t,
        ate=ate_flag,
        kernel_id=kernel_id,
        alpha=alpha,
    )

    # Convert to CPU numpy arrays
    S1_hat_np = S1_hat.detach().cpu().numpy()
    S0_hat_np = S0_hat.detach().cpu().numpy()
    diff_np = diff_hat.detach().cpu().numpy()
    var_diff_np = var_diff_t.detach().cpu().numpy()

    # SE and 95% CI for the survival difference
    se_np = np.sqrt(np.maximum(var_diff_np, 0.0))
    ci_lower_np = diff_np - 1.96 * se_np
    ci_upper_np = diff_np + 1.96 * se_np

    # -----------------------------
    # 8. Assemble result dict
    # -----------------------------
    result = {
        "propensity_scores": ps.tolist(),
        "total_loss": float(loss.item()),
        "max_lsd": float(final_LSD_max.item()),
        "mean_lsd": float(final_LSD_mean.item()),
        "times": t_grid.astype(float).tolist(),
        "S1": S1_hat_np.astype(float).tolist(),
        "S0": S0_hat_np.astype(float).tolist(),
        "surv_diff": diff_np.astype(float).tolist(),
        "var_diff": var_diff_np.astype(float).tolist(),
        "se": se_np.astype(float).tolist(),
        "ci_lower": ci_lower_np.astype(float).tolist(),
        "ci_upper": ci_upper_np.astype(float).tolist(),
    }

    print("✅ LBC-Net survival estimation completed successfully.")
    return result
