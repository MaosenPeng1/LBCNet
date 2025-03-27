import torch
import torch.nn as nn

class lbc_net(nn.Module):
    """
    LBC-Net: A neural network architecture for LBC-Net.

    This model consists of:
    - An initial input layer with batch normalization and ReLU activation.
    - Multiple hidden layers (L - 1) with batch normalization and ReLU activation.
    - A final output layer with a modified sigmoid activation to ensure overlap assumption.

    Attributes:
    ----------
    input_dim : int
        The number of input features (including intercept if added).
    hidden_dim : int, optional (default=100)
        The number of hidden units in each layer.
    L : int, optional (default=2)
        The number of total layers (excluding input and output).

    Methods:
    -------
    forward(x, epsilons=0.001)
        Computes the forward pass through the network.

    load_vae_encoder_weights(vae_encoder_weights)
        Loads pre-trained VAE encoder weights into the model.
    """

    def __init__(self, input_dim, hidden_dim=100, L=2, epsilon=0.001, dropout_rate=0.2):
        """
        Initialize the LBC-Net model.

        Parameters:
        ----------
        input_dim : int
            Number of input features.
        hidden_dim : int, optional (default=100)
            Number of hidden units in each layer.
        L : int, optional (default=2)
            Total number of layers (excluding input and output).
        """
        super(lbc_net, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.L = L
        self.dropout_rate = dropout_rate

        # Register epsilon as a PyTorch buffer (avoids computation overhead)
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32))

        # Initial input layer with batch normalization and ReLU activation
        self.initial_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.initial_bn = nn.BatchNorm1d(self.hidden_dim)
        self.initial_activation = nn.ReLU()
        self.initial_dropout = nn.Dropout(self.dropout_rate)
    
        # Define middle hidden layers (L-1 layers)
        self.middle_layers = nn.Sequential(
            *(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ) for _ in range(self.L - 1))
        )

        # Final output layer
        self.final_layer = nn.Linear(self.hidden_dim, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x, epsilons=0.001):
        """
        Forward pass through the LBC-Net.

        Parameters:
        ----------
        x : torch.Tensor
            Input feature matrix of shape (batch_size, input_dim).
        epsilons : float, optional (default=0.001)
            A small value to ensure overlap assumption.

        Returns:
        -------
        torch.Tensor
            Propensity scores scaled within (epsilons, 1 - epsilons).
        """
        # Pass through the initial layer
        x = self.initial_layer(x)
        x = self.initial_bn(x)
        x = self.initial_activation(x)
        x = self.initial_dropout(x)
        
        # Pass through middle layers
        for layer in self.middle_layers:
            x = layer(x)

        # Pass through the final layer
        x = self.final_layer(x)
        x = self.epsilon + (1 - 2 * self.epsilon) * self.output_activation(x)  # Modified sigmoid for overlap assumption

        return x

    def load_vae_encoder_weights(self, vae_encoder_weights):
        """
        Load weights from a pre-trained VAE encoder into the network.

        This method updates matching layers in the LBC-Net with the 
        corresponding weights from the VAE encoder.

        Parameters:
        ----------
        vae_encoder_weights : dict
            A dictionary containing pre-trained VAE encoder weights.
        """
        own_state = self.state_dict()
        encoder_state_dict = {k: v for k, v in vae_encoder_weights.items() if k in own_state}
        own_state.update(encoder_state_dict)
        self.load_state_dict(own_state)

def enable_dropout(model):
    """Enable dropout layers during inference (MC Dropout)."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def estimate_ate_att(Y, Tr, ps, ate=1):
    """
    Estimate ATE or ATT based on input.

    Parameters:
    -----------
    Y : np.ndarray
        Outcome variable, shape: [N]
    Tr : np.ndarray
        Treatment assignment (binary), shape: [N]
    ps : np.ndarray
        Propensity scores, shape: [N]
    ate : int, optional (default=1)
        If 1, estimate ATE. If 0, estimate ATT.

    Returns:
    --------
    treatment_effect : float
        Estimated ATE or ATT.
    """
    # Defensive programming: ensure all arrays are numpy arrays
    Y = np.asarray(Y)
    Tr = np.asarray(Tr)
    ps = np.asarray(ps)

    if ate == 1:
        # ATE weights
        wt = Tr / ps + (1 - Tr) / (1 - ps)

        # Weighted means
        ATE_treated = np.sum(Tr * wt * Y) / np.sum(Tr * wt)
        ATE_control_denom = np.sum((1 - Tr) * wt)

        if ATE_control_denom == 0:
            print("Warning: Control group sum is zero. Cannot compute ATE.")
            return np.nan

        ATE_control = np.sum((1 - Tr) * wt * Y) / ATE_control_denom
        return ATE_treated - ATE_control

    elif ate == 0:
        # ATT weights: weights on controls are ps / (1 - ps)
        wt = ps / (1 - ps)

        ATT_treated_denom = np.sum(Tr)
        ATT_control_denom = np.sum((1 - Tr) * wt)

        if ATT_control_denom == 0:
            print("Warning: Control group sum is zero. Cannot compute ATT.")
            return np.nan

        ATT_treated = np.sum(Tr * Y) / ATT_treated_denom
        ATT_control = np.sum((1 - Tr) * wt * Y) / ATT_control_denom

        return ATT_treated - ATT_control

    else:
        raise ValueError("Invalid 'ate' argument. Use 1 for ATE, 0 for ATT.")

def mc_dropout_inference_ate_att(model, Z, Y, Tr, ate=1, num_samples=100):
    """
    Perform MC Dropout inference and compute ATE/ATT estimates.

    Parameters:
    -----------
    model : nn.Module
        Trained LBC-Net model.
    Z : torch.Tensor
        Input data, shape: [N, input_dim]
    Y : np.ndarray
        Outcome variable, shape: [N]
    Tr : np.ndarray
        Treatment assignment, shape: [N]
    ate : int
        1 for ATE, 0 for ATT
    num_samples : int
        Number of stochastic forward passes.

    Returns:
    --------
    ps_point_estimates : np.ndarray
        Mean propensity scores across MC samples, shape: [N]
    ps_variances : np.ndarray
        Variance of propensity scores across MC samples, shape: [N]
    effect_point_estimate : float
        Mean ATE/ATT across MC samples.
    effect_variance : float
        Variance of ATE/ATT across MC samples.
    """
    model.eval()
    enable_dropout(model)

    all_ps_preds = []
    ate_att_estimates = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Forward pass with dropout active
            ps_preds = model(Z).squeeze().cpu().numpy()  # Propensity scores, shape: [N]
            all_ps_preds.append(ps_preds)

            # Compute ATE or ATT for this MC sample
            effect = estimate_ate_att(Y, Tr, ps_preds, ate=ate)
            ate_att_estimates.append(effect)

    # Convert predictions and effects to numpy arrays
    all_ps_preds = np.stack(all_ps_preds)  # Shape: [num_samples, N]
    ate_att_estimates = np.array(ate_att_estimates)  # Shape: [num_samples]

    # Propensity scores: mean and variance across MC samples
    ps_point_estimates = all_ps_preds.mean(axis=0)

    # Treatment effect: point estimate and variance
    effect_point_estimate = ate_att_estimates.mean()
    effect_variance = ate_att_estimates.var()

    return ps_point_estimates, effect_point_estimate, effect_variance

class vae(nn.Module):
    """
    Variational Autoencoder (VAE) for learning low-dimensional representations.

    This model consists of:
    - An encoder network that maps input data to a latent space.
    - A reparameterization trick to sample from the latent space.
    - A decoder network that reconstructs data from the latent representation.
    
    Attributes:
    ----------
    encoder : nn.Sequential
        The encoder network that outputs latent mean and log variance.
    decoder : nn.Sequential
        The decoder network that reconstructs the input from latent space.

    Methods:
    -------
    reparameterize(mu, logvar)
        Performs the reparameterization trick to enable backpropagation.
    forward(x)
        Passes input data through the VAE model, returning reconstructed data and latent variables.
    """

    def __init__(self, input_dim, latent_dim, hidden_dim=10):
        """
        Initialize the VAE model.

        Parameters:
        ----------
        input_dim : int
            Dimensionality of the input data.
        latent_dim : int
            Dimensionality of the latent space.
        hidden_dim : int, optional (default=10)
            Number of hidden units in the encoder and decoder.
        """
        super(vae, self).__init__()  

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Output mean and log variance
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick for sampling.

        Parameters:
        ----------
        mu : torch.Tensor
            The mean of the latent variables.
        logvar : torch.Tensor
            The log variance of the latent variables.

        Returns:
        -------
        torch.Tensor
            Sampled latent vector using the reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)  # Compute standard deviation
        eps = torch.randn_like(std)  # Random noise
        return mu + eps * std  # Sample from the latent distribution
      
    def forward(self, x):
        """
        Forward pass through the VAE.

        Parameters:
        ----------
        x : torch.Tensor
            Input data tensor of shape (batch_size, input_dim).

        Returns:
        -------
        torch.Tensor
            Reconstructed input data.
        torch.Tensor
            Mean of the latent variables.
        torch.Tensor
            Log variance of the latent variables.
        """
        latent_params = self.encoder(x)  # Compute latent parameters
        mu, logvar = torch.chunk(latent_params, 2, dim=1)  # Split into mean & log variance
        z = self.reparameterize(mu, logvar)  # Sample latent vector
        x_recon = self.decoder(z)  # Reconstruct input
        return x_recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """
    Computes the loss function for a Variational Autoencoder (VAE).

    The loss consists of:
    1. **Reconstruction Loss** (Mean Squared Error) - Measures how well the reconstructed data matches the original input.
    2. **KL Divergence** - Regularization term that enforces the latent distribution to be close to a standard normal distribution.

    Parameters:
    ----------
    recon_x : torch.Tensor
        Reconstructed input tensor (output of the decoder).
    x : torch.Tensor
        Original input tensor.
    mu : torch.Tensor
        Mean of the latent space distribution.
    logvar : torch.Tensor
        Log variance of the latent space distribution.

    Returns:
    -------
    torch.Tensor
        The total VAE loss (reconstruction loss + KL divergence).
    """
    
    # Reconstruction loss (how well the reconstructed output matches the input)
    reconstruction_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') 

    # KL Divergence (ensures latent distribution approximates standard normal)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 

    return reconstruction_loss + kl_divergence


def omega_calculate(propensity_scores, ck, h, kernel_id=0):
    """
    Computes the kernel weight function omega for a given set of propensity scores.

    This function calculates weights using different kernel functions to measure
    the density around given kernel centers (`ck`) with specified bandwidths (`h`).

    Parameters:
    ----------
    propensity_scores : torch.Tensor
        The propensity scores, a tensor of shape [N] (number of samples).
    ck : torch.Tensor
        The centers of the kernels, a tensor of shape [K] (number of kernel centers).
    h : torch.Tensor
        The bandwidths of the kernels, a tensor of shape [K].
    kernel_id : int, optional (default=0)
        The type of kernel function to use:
        - `0`: Gaussian kernel (default)
        - `1`: Uniform kernel
        - `2`: Epanechnikov kernel

    Returns:
    -------
    torch.Tensor
        The weight function omega, a tensor of shape [N, K] representing the computed
        kernel weights for each propensity score at each kernel center.
    """

    # Reshape propensity_scores to [N, 1] for broadcasting
    propensity_scores = propensity_scores.unsqueeze(1)  # Shape: [N, 1]

    # Compute normalized distance x for all propensity_scores and kernels (shape: [N, K])
    x = (propensity_scores - ck) / h

    # Compute kernel weights based on `kernel_id` instead of string comparison
    if kernel_id == 0:  # Gaussian kernel
        omega = (1 / torch.sqrt(2 * torch.tensor(torch.pi))) * torch.exp(-x**2 / 2)
    elif kernel_id == 1:  # Uniform kernel
        omega = 0.5 * (torch.abs(x) <= 1)  # Indicator function: 1 if |x| ≤ 1, else 0
    elif kernel_id == 2:  # Epanechnikov kernel
        omega = 0.75 * (1 - x**2) * (torch.abs(x) <= 1)  # Parabolic shape, supports [-1,1]
    else:
        raise ValueError(f"Invalid kernel ID: {kernel_id}. Choose from 0 (Gaussian), 1 (Uniform), or 2 (Epanechnikov).")

    # Apply bandwidth scaling (to adjust for different h values)
    return omega / h


def local_balance_ipw_loss(propensity_scores, treatment, Z, ck, h, w, ate=1):
    """
    Computes the local balance Inverse Probability Weighting (IPW) loss (Q1).

    This function calculates the imbalance of covariates across treatment groups
    using kernel-based weighting and inverse probability weighting.

    Parameters:
    ----------
    propensity_scores : torch.Tensor
        Estimated propensity scores, shape: [N].
    treatment : torch.Tensor
        Binary treatment assignments (0 or 1), shape: [N].
    Z : torch.Tensor
        Covariate matrix, shape: [N, p].
    ck : torch.Tensor
        Kernel centers, shape: [K].
    h : torch.Tensor
        Kernel bandwidths, shape: [K].
    w : torch.Tensor    
        Kernel weights, shape: [N, K].
    ipw : torch.Tensor
        Inverse Probability Weights (IPW), shape: [N].

    Returns:
    -------
    torch.Tensor
        The computed local balance IPW loss (scalar).
    """

    K = len(ck)  # Number of kernels

    # Compute IPW with ATT adjustment
    w_star = torch.ones_like(propensity_scores) if ate == 1 else propensity_scores # w*(p) = 1 (ATE) or w*(p) = p (ATT)

    ipw = treatment * propensity_scores + (1 - treatment) * (1 - propensity_scores) 
    W = (w * w_star.unsqueeze(1)) / ipw.unsqueeze(1)  # Expand for broadcasting (Shape: [N, K])

    # Compute `q` (covariate imbalance estimation) (Shape: [K, N, p])
    treatment_factor = (2 * treatment - 1).unsqueeze(1)  # Shape: [N, 1] -> Maps treatment: 1 → 1, 0 → -1
    q = treatment_factor * W  # Shape: [N, K]
    q = q.unsqueeze(2) * Z.unsqueeze(1)  # Shape: [N, K, p]
    q = q.permute(1, 0, 2)  # Reorder to [K, N, p]
    qvecT = torch.sum(q, dim=1)  # Sum over N → Shape: [K, p]

    # Compute Sigma matrices for all kernels
    temp0 = ((w**2) * (w_star.unsqueeze(1)**2)).unsqueeze(2) * Z.unsqueeze(1)  # Shape: [N, K, p]
    temp0 = temp0.permute(1, 0, 2)  # Shape: [K, N, p]
    Z_t = Z.transpose(0, 1).unsqueeze(0).expand(K, -1, -1)  # Shape: [K, p, N]
    temp1 = torch.bmm(Z_t, temp0)  # Batched matrix multiplication (Shape: [K, p, p])
    sigma = temp1 / (ck * (1 - ck)).view(K, 1, 1)  # Shape: [K, p, p]

    # Compute pseudo-inverse of Sigma (Shape: [K, p, p])
    sigma_inv = torch.linalg.pinv(sigma)

    # Compute loss for each kernel
    A = torch.bmm(qvecT.unsqueeze(1), sigma_inv)  # Shape: [K, 1, p]
    loss_per_kernel = torch.bmm(A, qvecT.unsqueeze(2))  # Shape: [K, 1, 1]

    # Aggregate loss across valid kernels
    valid_kernels = torch.sum(w, dim=0) > 0  # Identify valid kernels
    K_new = torch.sum(valid_kernels).item()  # Count valid kernels
    loss = torch.sum(loss_per_kernel[valid_kernels]) / K_new  # Normalize loss

    return loss

def penalty_loss(propensity_scores, treatment, ck, h, w):
    """
    Computes the penalty loss term (Q2) for propensity score estimation.

    This function calculates the local penalty for imbalance using kernel-weighted 
    squared differences between the treatment assignment and estimated propensity scores.

    Parameters:
    ----------
    propensity_scores : torch.Tensor
        Estimated propensity scores, shape: [N].
    treatment : torch.Tensor
        Binary treatment assignments (0 or 1), shape: [N].
    ck : torch.Tensor
        Kernel centers, shape: [K].
    h : torch.Tensor
        Kernel bandwidths, shape: [K].
    w : torch.Tensor
        Kernel weights, shape: [N, K].

    Returns:
    -------
    torch.Tensor
        The calculated penalty loss term (scalar).
    """

    # Compute numerator: Weighted squared differences (Shape: [K])
    diff_squared = (treatment - propensity_scores) ** 2  # Shape: [N]
    numerator = torch.sum(w * diff_squared.unsqueeze(1), dim=0)  # Shape: [K]

    # Compute denominator: Weighted sum scaled by ck(1 - ck) (Shape: [K])
    sum_w = torch.sum(w, dim=0)  # Shape: [K]
    denominator = torch.where(
        sum_w == 0,
        torch.tensor(1.0, dtype=torch.float32),  # Avoid division by zero
        ck * (1 - ck) * sum_w
    )  # Shape: [K]

    # Compute kernel-wise loss (Shape: [K])
    kernel_loss = numerator / denominator

    # Compute final loss by averaging over valid kernels (where sum_w > 0)
    valid_kernels = sum_w > 0  # Boolean mask for valid kernels
    loss = torch.sum(kernel_loss[valid_kernels]) / torch.sum(valid_kernels)

    return loss


def lsd_cal(propensity_scores, treatment, Z, ck, h, kernel_id, ate=1):
    """
    Computes the Local Standardized Difference (LSD) between treated and control groups.

    This function calculates LSD values using kernel-weighted inverse probability 
    weighting (IPW), which measures covariate balance across treatment groups.

    Parameters:
    ----------
    propensity_scores : torch.Tensor
        Estimated propensity scores, shape: [N].
    treatment : torch.Tensor
        Binary treatment assignments (0 for control, 1 for treatment), shape: [N].
    Z : torch.Tensor
        Covariate matrix including an intercept column, shape: [N, p].
    ck : torch.Tensor
        Kernel centers, shape: [K].
    h : torch.Tensor
        Kernel bandwidths, shape: [K].
    kernel_id : int
        The type of kernel function to use:
        - `0`: Gaussian kernel
        - `1`: Uniform kernel
        - `2`: Epanechnikov kernel
    ate : int, optional (default=1)
        The Average Treatment Effect (ATE) adjustment for IPW. Set to 1 for ATE, 0 for ATT.    

    Returns:
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - `LSD_max`: Maximum absolute LSD value (scalar).
        - `LSD_mean`: Mean absolute LSD value (scalar).
    """

    w_star = torch.ones_like(propensity_scores) if ate == 1 else propensity_scores # w*(p) = 1 (ATE) or w*(p) = p (ATT)
    w = omega_calculate(propensity_scores, ck, h, kernel_id)
    ipw = treatment * propensity_scores + (1 - treatment) * (1 - propensity_scores) 
    W = (w * w_star.unsqueeze(1)) / ipw.unsqueeze(1)  # Expand for broadcasting (Shape: [N, K])

    # Compute Treatment & Control Weights (Shape: [N, K])
    treatment_W = treatment.unsqueeze(1) * W  # Shape: [N, K]
    control_W = (1 - treatment).unsqueeze(1) * W  # Shape: [N, K]

    # Compute Mean Covariate Values for Treatment & Control Groups (Shape: [K, p])
    mu1 = torch.sum(treatment_W.unsqueeze(2) * Z.unsqueeze(1), dim=0) / \
          torch.sum(treatment_W, dim=0, keepdim=True).transpose(0, 1)
    mu0 = torch.sum(control_W.unsqueeze(2) * Z.unsqueeze(1), dim=0) / \
          torch.sum(control_W, dim=0, keepdim=True).transpose(0, 1)

    # Compute Variance for Treatment & Control Groups (Shape: [K, p])
    v1 = torch.sum(treatment_W.transpose(0, 1).unsqueeze(2) * (Z.unsqueeze(0) - mu1.unsqueeze(1))**2, dim=1) / \
         torch.sum(treatment_W, dim=0, keepdim=True).transpose(0, 1)
    v0 = torch.sum(control_W.transpose(0, 1).unsqueeze(2) * (Z.unsqueeze(0) - mu0.unsqueeze(1))**2, dim=1) / \
         torch.sum(control_W, dim=0, keepdim=True).transpose(0, 1)

    # Apply clamping to avoid zero variance (minimal effect)
    v1 = torch.clamp(v1, min=1e-8)
    v0 = torch.clamp(v0, min=1e-8)

    # Compute Effective Sample Sizes (ESS) (Shape: [K])
    ess1 = (torch.sum(treatment_W, dim=0)**2) / torch.sum(treatment_W**2, dim=0)
    ess0 = (torch.sum(control_W, dim=0)**2) / torch.sum(control_W**2, dim=0)

    # Compute LSD (Shape: [K, p])
    LSD = 100 * (mu1 - mu0) / torch.sqrt((ess1.unsqueeze(1) * v1 + ess0.unsqueeze(1) * v0) / (ess1.unsqueeze(1) + ess0.unsqueeze(1)))

    # Aggregate LSD Results
    LSD_mean = torch.mean(torch.abs(LSD))  # Mean absolute LSD
    LSD_max = torch.max(torch.abs(LSD))    # Maximum absolute LSD

    return LSD_max, LSD_mean
