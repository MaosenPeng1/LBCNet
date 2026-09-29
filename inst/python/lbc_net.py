import copy
import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bar
import time  # Import time for execution tracking

try:
    import torch
except ImportError:
    raise ImportError("🚨 Error: 'torch' is not installed. Install it using: pip install torch")

try:
    import numpy as np
except ImportError:
    raise ImportError("🚨 Error: 'numpy' is not installed. Install it using: pip install numpy")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the `inst/python/` directory to the system path
sys.path.append(script_dir)

from lbc_helpers import *

def run_lbc_net(data_df, Z_columns, T_column, Y_column, estimand, ck, h, 
                kernel = "gaussian", gpu=0, ate = 1,
                seed=100, hidden_dim=100, L=2, 
                vae_epochs=250, vae_lr=0.01, 
                max_epochs=5000, lr=0.05, weight_decay=1e-5, 
                balance_lambda=1.0, epsilon = 0.001, lsd_threshold=2, alpha = 0.01,
                rolling_window=5, show_progress=True, compute_variance=True):
    """
    Runs the LBC-Net estimation for propensity score calculation.

    This function trains a Variational Autoencoder (VAE) to learn latent representations 
    of covariates and then uses an LBC-Net model to estimate propensity scores. It applies 
    kernel-based local balance adjustments to improve covariate balance.

    If `Y_column` is provided, also computes an IPW estimand (ATE / ATT / mu1 / mu0)
    and, if `compute_variance=True` and estimand != "Y", an IF-based SE and CI.

    Parameters:
    ----------
    data_df : pandas.DataFrame
        Dataset containing treatment assignment and covariates.
    Z_columns : list of str
        Names of covariate columns.
    T_column : str
        Name of the treatment assignment column.
    Y_column : str or None, default None
        Name of the outcome column in `data_df`. If None, only PS are estimated.
    estimand : {"ATE", "ATT", "Y"}, default "ATE"
        Target estimand when Y_column is not None.
        - "ATE": Average Treatment Effect.
        - "ATT": Average Treatment Effect on the Treated.
        - "mu1"  : Weighted mean outcome among treated.
        - "mu0"  : Weighted mean outcome among control.
    ck : list or numpy.ndarray
        Kernel center values for balance adjustment.
    h : list or numpy.ndarray
        Kernel bandwidth values for weighting.
    kernel : str, optional (default="gaussian")
        Kernel function for local balance adjustment.
        Supported values: ["gaussian", "epanechnikov", "uniform"].
    epsilon : float, optional (default=0.001)
        Epsilon value for numerical stability in kernel computation.
    ate : float, optional (default=1)
        Average Treatment Effect (ATE) for balancing.
        If `ate=1`, the propensity score aims for ATE is estimated from the data. 
        If `ate=0`, the propensity score aims for ATT is estimated from the data.

    GPU & Reproducibility:
    ----------------------
    gpu : int, optional (default=0)
        GPU device ID (if using CUDA).
    seed : int, optional (default=100)
        Random seed for reproducibility in PyTorch.

    Network Architecture & Training:
    --------------------------------
    hidden_dim : int, optional (default=100)
        Number of hidden units in the LBC-Net.
    L : int, optional (default=2)
        Number of hidden layers in the LBC-Net.
    vae_epochs : int, optional (default=250)
        Number of epochs for training the VAE.
    vae_lr : float, optional (default=0.01)
        Learning rate for the VAE optimizer.
    max_epochs : int, optional (default=5000)
        Maximum number of epochs for training the LBC-Net.
    lr : float, optional (default=0.05)
        Learning rate for the LBC-Net optimizer.
    weight_decay : float, optional (default=1e-5)
        L2 regularization (weight decay) for optimizer.
    alpha : float, optional (default=0.01)
        Small ridge penalty factor for stabilizing the chain correction.

    Stopping Criteria:
    ------------------
    balance_lambda : float, optional (default=1.0)
        Weight for the penalty loss term in training.
    lsd_threshold : float, optional (default=2)
        Threshold for stopping criteria based on LSD (Local Standardized Difference).
    rolling_window : int, optional (default=5)
        Number of past LSD values considered for early stopping.
    show_progress : bool, optional (default=True)
        Display progress bar for training epochs.
    compute_variance : bool, optional (default=True)
        Whether to compute IF-based standard errors and confidence intervals.

    Returns:
    -------
    dict
        A dictionary containing:
        - `"propensity_scores"`: List of estimated propensity scores.
        - `"total_loss"`: Total loss value.
        - `"max_lsd"`: Maximum LSD value.
        - `"mean_lsd"`: Mean LSD value

        If Y_column is not None, also includes:
            - "effect": float
            - "se": float or None
            - "ci_lower": float or None
            - "ci_upper": float or None
    """

    # Set Device for Computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu))

    # Set Random Seed for Reproducibility
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True

    # Convert `ck` and `h` into PyTorch tensors
    # Ensure ck and h are always at least 1D tensors
    if not isinstance(ck, (list, tuple)):
        ck = [ck]
    if not isinstance(h, (list, tuple)):
        h = [h]

    ck = torch.tensor(ck, dtype=torch.float32).to(device)
    h = torch.tensor(h, dtype=torch.float32).to(device)


    # Convert DataFrame to Tensors
    Z_numpy = data_df.loc[:, Z_columns].to_numpy(dtype="float32")  # Explicit NumPy conversion
    Z = torch.tensor(Z_numpy, dtype=torch.float32, device=device)

    T_numpy = data_df.loc[:, T_column].to_numpy(dtype="float32")  # Explicit NumPy conversion
    T = torch.tensor(T_numpy, dtype=torch.float32, device=device)

    has_outcome = (Y_column is not None) and (Y_column in data_df.columns)
    Y = None
    if has_outcome:
        Y_numpy = data_df.loc[:, Y_column].to_numpy(dtype="float32")
        Y = torch.tensor(Y_numpy, dtype=torch.float32, device=device)

    n, p = Z.shape  # Number of samples (N) and covariates (p)

    # Normalize Covariates (Z)
    Z_norm = (Z - Z.mean(dim=0)) / Z.std(dim=0)
    Z_norm = torch.cat([torch.ones((n, 1), device=device), Z_norm], dim=1) # Add intercept
    p += 1 # Adjust for intercept

    kernel_id = {"gaussian": 0, "uniform": 1, "epanechnikov": 2}[kernel]

    # Train Variational Autoencoder (VAE)
    vae_model = vae(p, p).to(device)
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=vae_lr)
    
    vae_model.train()
    for epoch in range(int(vae_epochs)):
        vae_optimizer.zero_grad()
        recon_batch, mu, logvar = vae_model(Z_norm)
        loss = vae_loss(recon_batch, Z_norm, mu, logvar)
        loss.backward()
        vae_optimizer.step()

    # Train LBC-Net Model
    ps_model = lbc_net(p, hidden_dim, L, epsilon).to(device)
    optimizer = optim.Adam(ps_model.parameters(), lr=lr, weight_decay=0.0)
    ps_model.load_vae_encoder_weights(vae_model.encoder.state_dict())

    # LSD early stopping window
    lsd_window = []  
    
    # Track whether early stopping happened
    early_stopping = False  
    phase1_state = None
    phase1_epoch = None
    phase1_lsd_max = None
    phase1_lsd_mean = None

    # Initialize Progress Bar if `show_progress=True`
    if show_progress:
        pbar = tqdm(
            total=max_epochs, 
            desc="Training Progress", 
            position=0, 
            leave=True,
            bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{rate_fmt} {postfix}]"
        )

        start_time = time.time()  # Start timing

    for epoch in range(int(max_epochs)):
        ps_model.train()
        optimizer.zero_grad()
        outputs = ps_model(Z_norm).squeeze()

        loss = lbc_net_loss(outputs, T, Z_norm, ck, h, ate=ate, kernel_id=kernel_id, balance_lambda = balance_lambda)

        loss.backward()
        optimizer.step()

        # Update Progress Bar (if enabled)
        if show_progress:
            elapsed_time = time.time() - start_time  # Time elapsed so far
            avg_time_per_epoch = elapsed_time / (epoch + 1)  # Average time per epoch
            estimated_total_time = avg_time_per_epoch * max_epochs  # Total estimated time
            remaining_time = max(0, estimated_total_time - elapsed_time)  # Ensure non-negative

            # Correct the comparison display order (Elapsed Time first)
            pbar.set_postfix({
                "Remaining Time (s)": f"{remaining_time:.2f}",
                "Elapsed Time (s)": f"{elapsed_time:.2f}",
                "Loss": f"{loss.item():.4f}"
            })
            pbar.update(1)

        # Early Stopping Based on LSD Threshold
        if (epoch + 1) % 200 == 0:
            ps_model.eval()
            with torch.no_grad():
                LSD_max, LSD_mean = lsd_cal(ps_model(Z_norm).squeeze(), T, Z, ck, h, kernel_id, ate = ate)
                lsd_window.append(LSD_max)

                # Maintain the rolling window size
                if len(lsd_window) > rolling_window:
                    lsd_window.pop(0)

                # Compute rolling LSD mean and stop if below threshold
                if len(lsd_window) == rolling_window:
                    mean_lsd_window = torch.mean(torch.stack(lsd_window))
                    if mean_lsd_window < lsd_threshold:
                        phase1_state = copy.deepcopy(ps_model.state_dict())
                        phase1_epoch = epoch + 1
                        phase1_lsd_max = float(LSD_max.detach().cpu().item())
                        phase1_lsd_mean = float(LSD_mean.detach().cpu().item())
                        print(f"✅ Stopping early at epoch {epoch + 1} (rolling average max LSD < {lsd_threshold}%)")
                        early_stopping = True
                        break

    if not early_stopping:
        print("⚠️ Stopping criterion not met at max epochs. "
            "Try increasing `max_epochs` or adjusting `lsd_threshold` for better convergence.")  

    phase2_selected = False
    phase2_pbar = None
    if show_progress and pbar is not None:
        pbar.close()

    if early_stopping and phase1_state is not None:
        phase2_lr = lr * 0.1
        phase2_optimizer = optim.Adam(ps_model.parameters(), lr=phase2_lr, weight_decay=0.0)
        phase2_check_interval = 100
        phase2_max_epochs = 3000
        phase2_rel_tol = 1e-3
        phase2_abs_tol = 1e-10
        phase2_patience = 5
        phase2_best_loss = None
        phase2_best_epoch = None
        phase2_best_state = None
        phase2_previous_loss = None
        phase2_consecutive = 0

        if show_progress:
            phase2_pbar = tqdm(
                total=phase2_max_epochs,
                desc="Phase 2 refinement",
                position=0,
                leave=False,
                bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{rate_fmt} {postfix}]"
            )

        for phase2_epoch in range(1, phase2_max_epochs + 1):
            ps_model.train()
            phase2_optimizer.zero_grad()
            phase2_outputs = ps_model(Z_norm).squeeze()
            phase2_loss = lbc_net_loss(
                phase2_outputs, T, Z_norm, ck, h, ate=ate,
                kernel_id=kernel_id, balance_lambda=balance_lambda
            )
            phase2_loss.backward()
            phase2_optimizer.step()

            if show_progress and phase2_pbar is not None:
                phase2_pbar.set_postfix({
                    "Phase": "2",
                    "Epoch": phase2_epoch,
                    "Loss": f"{phase2_loss.item():.4f}"
                })
                phase2_pbar.update(1)

            if phase2_epoch % phase2_check_interval == 0:
                ps_model.eval()
                with torch.no_grad():
                    checked_outputs = ps_model(Z_norm).squeeze()
                    current_loss = lbc_net_loss(
                        checked_outputs, T, Z_norm, ck, h, ate=ate,
                        kernel_id=kernel_id, balance_lambda=balance_lambda
                    )
                    current_loss_value = float(current_loss.detach().cpu().item())

                if (phase2_best_loss is None) or (current_loss_value < phase2_best_loss):
                    phase2_best_loss = current_loss_value
                    phase2_best_epoch = phase2_epoch
                    phase2_best_state = copy.deepcopy(ps_model.state_dict())

                if phase2_previous_loss is not None:
                    loss_change = abs(current_loss_value - phase2_previous_loss)
                    loss_tolerance = phase2_abs_tol + phase2_rel_tol * max(
                        abs(current_loss_value), abs(phase2_previous_loss)
                    )
                    if loss_change <= loss_tolerance:
                        phase2_consecutive += 1
                    else:
                        phase2_consecutive = 0

                    if phase2_consecutive >= phase2_patience:
                        break

                phase2_previous_loss = current_loss_value

        if phase2_best_state is not None:
            ps_model.load_state_dict(phase2_best_state)
            with torch.no_grad():
                selected_outputs = ps_model(Z_norm).squeeze()
                selected_lsd_max, selected_lsd_mean = lsd_cal(
                    selected_outputs, T, Z, ck, h, kernel_id, ate=ate
                )
                final_lsd_max = float(selected_lsd_max.detach().cpu().item())
                final_lsd_mean = float(selected_lsd_mean.detach().cpu().item())
                final_loss_value = float(
                    lbc_net_loss(
                        selected_outputs, T, Z_norm, ck, h, ate=ate,
                        kernel_id=kernel_id, balance_lambda=balance_lambda
                    ).detach().cpu().item()
                )

            if torch.isfinite(torch.tensor(final_loss_value)) and final_lsd_max < lsd_threshold:
                phase2_selected = True
            else:
                ps_model.load_state_dict(phase1_state)
                phase2_selected = False
                print("⚠️ Phase 2 refinement did not preserve the original Phase 1 balance criterion; restoring the Phase 1 model.")
        else:
            ps_model.load_state_dict(phase1_state)

        if show_progress and phase2_pbar is not None:
            phase2_pbar.close()

    # Close Progress Bar if enabled
    if show_progress and pbar is not None:
        pbar.close() 

    # Compute Final Propensity Scores
    with torch.no_grad():
        final_outputs = ps_model(Z_norm).squeeze()
        final_LSD_max, final_LSD_mean = lsd_cal(final_outputs, T, Z, ck, h, kernel_id, ate = ate)
        ps = final_outputs.detach().cpu().numpy()

    result = {
        "propensity_scores": ps.tolist(),
        "total_loss": float(loss.item()),
        "max_lsd": float(final_LSD_max.item()),
        "mean_lsd": float(final_LSD_mean.item()),
    }

    # -----------------------
    # 7. Effect + variance (if Y is available)
    # -----------------------
    if has_outcome:
        print("Starting post-processing: computing treatment effect and variance...")

        effect = None
        se_val = None
        ci_lower = None
        ci_upper = None

        # Always get the plug-in IPW estimate (ATE / ATT / mu1/ mu0)
        with torch.no_grad():
            theta_hat = ipw_est(Y, T, final_outputs, estimand=estimand)
            effect = float(theta_hat.detach().cpu().item())

            # Plug-in IF treating PS as fixed (for this estimand)
            phi_ipw = plug_in_if(Y, T, final_outputs, estimand=estimand)

            # Provide treatment-specific mean estimates as expected by the R wrapper.
            mu1 = ipw_est(Y, T, final_outputs, estimand="mu1")
            mu0 = ipw_est(Y, T, final_outputs, estimand="mu0")
            means = torch.stack([mu1, mu0], dim=0)

            result["means"] = means.detach().cpu().numpy().tolist()

        # IF-based SE and CI only if requested 
        if compute_variance:
            joint = if_var(
                ps_model,
                T,
                Y,
                Z_norm,
                ck,
                h,
                phi_ipw,
                ate=ate,
                estimand=estimand,    
                kernel_id=kernel_id,
                balance_lambda=balance_lambda,
                alpha=alpha,
                return_joint=True,
            )
            se_t = joint["se"]
            for component in (
                "means", "se_means", "covariance_means", "influence_functions"
            ):
                result[component] = joint[component].detach().cpu().numpy()
            # se_t may already be scalar; convert robustly
            se_val = float(
                se_t.detach().cpu().item() if hasattr(se_t, "detach") else se_t
            )
            ci_lower = effect - 1.96 * se_val
            ci_upper = effect + 1.96 * se_val

        # Attach to result
        result["effect"] = effect
        result["se"] = se_val
        result["ci_lower"] = ci_lower
        result["ci_upper"] = ci_upper
    
    print("✅ LBC-Net training completed successfully.")

    # Return Results
    return result
