import os
import sys
import torch
import torch.optim as optim
import numpy as np
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

def run_lbc_net(data_df, Z_columns, T_column, ck, h, 
                kernel = "gaussian", gpu=0, ate = 1,
                seed=100, hidden_dim=100, L=2, 
                vae_epochs=250, vae_lr=0.01, 
                max_epochs=5000, lr=0.05, weight_decay=1e-5, 
                balance_lambda=1.0, epsilon = 0.001, lsd_threshold=2, 
                rolling_window=5, show_progress=True):
    """
    Runs the LBC-Net estimation for propensity score calculation.

    This function trains a Variational Autoencoder (VAE) to learn latent representations 
    of covariates and then uses an LBC-Net model to estimate propensity scores. It applies 
    kernel-based local balance adjustments to improve covariate balance.

    Parameters:
    ----------
    data_df : pandas.DataFrame
        Dataset containing treatment assignment and covariates.
    Z_columns : list of str
        Names of covariate columns.
    T_column : str
        Name of the treatment assignment column.
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

    Returns:
    -------
    dict
        A dictionary containing:
        - `"propensity_scores"`: List of estimated propensity scores.
        - `"balance_loss"`: Value of the balance loss term.
        - `"calibration_loss"`: Value of the calibration loss term.
        - `"total_loss"`: Total loss value.
        - `"max_lsd"`: Maximum LSD value.
        - `"mean_lsd"`: Mean LSD value
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
    ck = torch.tensor(ck, dtype=torch.float32).to(device)
    h = torch.tensor(h, dtype=torch.float32).to(device)

    # Convert DataFrame to Tensors
    Z = torch.tensor(data_df[Z_columns].values, dtype=torch.float32, device=device)
    T = torch.tensor(data_df[T_column].values, dtype=torch.float32, device=device)
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
    optimizer = optim.Adam(ps_model.parameters(), lr=lr, weight_decay=weight_decay)
    ps_model.load_vae_encoder_weights(vae_model.encoder.state_dict())

    # LSD early stopping window
    lsd_window = []  
    
    # Track whether early stopping happened
    early_stopping = False  

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

    e = 1e-6  # Small constant for numerical stability
    for epoch in range(int(max_epochs)):
        ps_model.train()
        optimizer.zero_grad()
        outputs = ps_model(Z_norm).squeeze()

        # Compute loss components
        w = omega_calculate(outputs, ck, h, kernel_id)
        w_stable = torch.where((torch.abs(w) < e) & (w != 0), torch.full_like(w, e), w)
        
        penalty = penalty_loss(outputs, T, ck, h, w_stable)
        balance_loss = local_balance_ipw_loss(outputs, T, Z_norm, ck, h, w_stable, ate)
        loss = balance_lambda * penalty + balance_loss

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
                LSD_max, LSD_mean = lsd_cal(ps_model(Z_norm).squeeze(), T, Z, ck, h, kernel_id, ate)
                lsd_window.append(LSD_max)

                # Maintain the rolling window size
                if len(lsd_window) > rolling_window:
                    lsd_window.pop(0)

                # Compute rolling LSD mean and stop if below threshold
                if len(lsd_window) == rolling_window:
                    mean_lsd_window = torch.mean(torch.stack(lsd_window))
                    if mean_lsd_window < lsd_threshold:
                        print(f"✅ Stopping early at epoch {epoch + 1} (rolling average max LSD < {lsd_threshold}%)")
                        early_stopping = True
                        break

    if not early_stopping:
        print("⚠️ Stopping criterion not met at max epochs. "
            "Try increasing `max_epochs` or adjusting `lsd_threshold` for better convergence.")  

    # Close Progress Bar if enabled
    if show_progress:
        pbar.close() 

    # Compute Final Propensity Scores
    with torch.no_grad():
        final_outputs = ps_model(Z_norm).squeeze()
        final_LSD_max, final_LSD_mean = lsd_cal(final_outputs, T, Z, ck, h, kernel_id, ate)
        ps = final_outputs.detach().cpu().numpy()

    print("✅ LBC-Net training completed successfully.")

    # Return Results
    return {
    "propensity_scores": ps.tolist(),
    "balance_loss": balance_loss.item(),  
    "calibration_loss": penalty.item(),   
    "total_loss": loss.item(),
    "max_lsd": final_LSD_max.item(),      
    "mean_lsd": final_LSD_mean.item()            
}
