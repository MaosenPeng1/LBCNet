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

    def __init__(self, input_dim, hidden_dim=100, L=2, epsilon=0.001):
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

        # Register epsilon as a PyTorch buffer (avoids computation overhead)
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32))

        # Initial input layer with batch normalization and ReLU activation
        self.initial_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.initial_bn = nn.BatchNorm1d(self.hidden_dim)
        self.initial_activation = nn.ReLU()
    
        # Define middle hidden layers (L-1 layers)
        self.middle_layers = nn.Sequential(
            *(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU()
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

def lbc_net_loss(propensity_scores, treatment, Z, ck, h, ate=1, kernel_id=0, balance_lambda =1.0):
    """
    Unified LBC-Net loss combining local balance and calibration moments.

    Parameters
    ----------
    propensity_scores : torch.Tensor, shape [N]
        Estimated propensity scores.
    treatment : torch.Tensor, shape [N]
        Binary treatment indicator (0/1).
    Z : torch.Tensor, shape [N, p]
        Covariate matrix (often Z_norm with intercept).
    ck : torch.Tensor, shape [K]
        Kernel centers.
    h : torch.Tensor, shape [K]
        Bandwidths.
    ate : int, default=1
        1 for ATE target, 0 for ATT target.
    kernel_id : int, default=0
        0 = Gaussian, 1 = Uniform, 2 = Epanechnikov.

    Returns
    -------
    torch.Tensor (scalar)
        The loss Q = E_k [ ||(B_k, C_k)||^2 ].
    """
    tiny = 1e-6

    # Kernel weights: shape [N, K]
    kernel_w = omega_calculate(propensity_scores, ck, h, kernel_id)
    kernel_w = torch.where(
        (torch.abs(kernel_w) < tiny) & (kernel_w != 0),
        torch.full_like(kernel_w, tiny),
        kernel_w
    )

    K = len(ck)          # number of kernels
    N, p = Z.shape       # N samples, p covariates

    # w*(p) = 1 (ATE) or w*(p) = p (ATT)
    w_star = torch.ones_like(propensity_scores) if ate == 1 else propensity_scores
    # Shape [N, K]
    w = kernel_w * w_star.unsqueeze(1)

    # d = P(A=a | X) under the “observed” treatment
    d = treatment * propensity_scores + (1 - treatment) * (1 - propensity_scores)

    # Balance moment: B_k = sum_i w_ik * ((2A_i - 1)/d_i) * Z_i
    V = ((2 * treatment - 1) / d).unsqueeze(1) * Z          # [N, p]
    B = w.transpose(0, 1) @ V                               # [K, p]

    # Calibration moment: C_k = sum_i w_ik * (A_i - p_i) / {ck_k (1 - ck_k)}
    C = (w.transpose(0, 1) @ (treatment - propensity_scores)) / (ck * (1 - ck))  # [K]

    # Stack [B_k, C_k] into D_k ∈ R^{p+1}
    C_scaled = balance_lambda * C
    D = torch.cat([B, C_scaled.unsqueeze(1)], dim=1) # [K, p+1]

    # Q = mean_k ||D_k||^2
    Q = (D * D).sum(dim=1).mean()

    return Q

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

def ipw_est(Y, T, ps, estimand="ATE"):
    """
    Compute IPW estimands: ATE, ATT, or Y (weighted treated mean).

    Parameters
    ----------
    Y : torch.Tensor, shape (n,)
        Outcome vector.
    T : torch.Tensor, shape (n,)
        Treatment assignment (0/1).
    ps : torch.Tensor, shape (n,)
        Propensity scores estimated by LBC-Net.
    estimand : {"ATE","ATT","mu1", "mu0"}
        Target causal estimand.

    Returns
    -------
    tau : torch.Tensor (scalar)
        Estimated causal estimand.
    """

    Y = Y.double()
    T = T.double()
    ps = ps.double()

    # denominator: d_i = T*p + (1-T)*(1-p)
    d = T * ps + (1 - T) * (1 - ps)

    # --------- ATE ---------
    if estimand == "ATE":
        # frequency weight: w* = 1
        w_star = torch.ones_like(ps)
        wt = w_star / d

        # treated mean
        num_treat = torch.sum(T * wt * Y)
        den_treat = torch.sum(T * wt)

        # control mean
        num_ctrl = torch.sum((1 - T) * wt * Y)
        den_ctrl = torch.sum((1 - T) * wt)

        if den_ctrl.item() == 0:
            return torch.tensor(float('nan'), device=Y.device)

        tau = (num_treat / den_treat) - (num_ctrl / den_ctrl)
        return tau

    # --------- ATT ---------
    elif estimand == "ATT":
        # treated mean (unweighted)
        num_treat = torch.sum(T * Y)
        den_treat = torch.sum(T)

        # ATT frequency weight: w* = p
        w_star = ps
        wt = w_star / d

        # reweighted control mean
        num_ctrl = torch.sum((1 - T) * wt * Y)
        den_ctrl = torch.sum((1 - T) * wt)

        if den_ctrl.item() == 0:
            return torch.tensor(float('nan'), device=Y.device)

        tau = (num_treat / den_treat) - (num_ctrl / den_ctrl)
        return tau
    
    # --------- mu1: mean potential outcome under treatment ---------
    elif estimand == "mu1":
        # IPW weights: w_i = 1 / d_i, and among treated d_i = p_i
        wt = 1.0 / d

        num = torch.sum(T * wt * Y)
        den = torch.sum(T * wt)

        if den.item() == 0:
            return torch.tensor(float('nan'), device=Y.device)

        mu1 = num / den
        return mu1


    # --------- mu0: mean potential outcome under control ---------
    elif estimand == "mu0":
        # IPW weights: w_i = 1 / d_i, and among controls d_i = 1 - p_i
        wt = 1.0 / d

        num = torch.sum((1 - T) * wt * Y)
        den = torch.sum((1 - T) * wt)

        if den.item() == 0:
            return torch.tensor(float('nan'), device=Y.device)

        mu0 = num / den
        return mu0

    else:
        raise ValueError(f"Unknown estimand '{estimand}'. Use 'ATE', 'ATT', 'mu0' or 'mu1'.")


def plug_in_if(Y, T, p, estimand="ATE"):
    """
    Plug-in IPW influence function treating p as known (no PS uncertainty).

    Supports three estimands:
      - ATE: μ1 - μ0 using Hájek IPW with weights T/p and (1-T)/(1-p)
      - ATT: E[Y | T=1] - μ0,ATT where μ0,ATT is a Hájek reweighted
             control mean with weights proportional to p/(1-p)
      - mu1  : weighted mean outcome among treated using IPW weights
             based on the fitted propensity scores.
      - mu0  : weighted mean outcome among control using IPW weights
             based on the fitted propensity scores.

    Parameters
    ----------
    Y : torch.Tensor, shape (n,)
        Outcome vector.
    T : torch.Tensor, shape (n,)
        Treatment indicator (0/1).
    p : torch.Tensor, shape (n,)
        Propensity scores, treated as fixed (plug-in).
    estimand : {"ATE", "ATT", "mu1", "mu0"}, default "ATE"
        Target estimand.

    Returns
    -------
    phi : torch.Tensor, shape (n,)
        Per-observation plug-in influence function values.
    """
    Y = Y.double()
    T = T.double()
    p = p.double()
    n = Y.numel()

    # -------- ATE: standard Hájek IPW μ1 - μ0 --------
    if estimand == "ATE":
        # Treated: a1 = T/p, b1 = T*Y/p
        a1 = T / p
        b1 = T * Y / p
        A1 = a1.sum()
        B1 = b1.sum()
        mu1 = B1 / A1

        # Control: a0 = (1-T)/(1-p), b0 = (1-T)*Y/(1-p)
        a0 = (1.0 - T) / (1.0 - p)
        b0 = (1.0 - T) * Y / (1.0 - p)
        A0 = a0.sum()
        B0 = b0.sum()
        mu0 = B0 / A0

        # Plug-in IFs
        denom1 = A1 / n  # ~ E[a1]
        denom0 = A0 / n  # ~ E[a0]
        phi1 = (b1 - mu1 * a1) / denom1
        phi0 = (b0 - mu0 * a0) / denom0

        phi = phi1 - phi0

    # -------- ATT: mean treated - reweighted control mean --------
    elif estimand == "ATT":
        # Treated mean μ1 = E[Y | T=1]
        a1 = T               # weights for treated
        b1 = T * Y
        A1 = a1.sum()
        B1 = b1.sum()
        mu1 = B1 / A1

        # Control mean μ0,ATT with weights proportional to p/(1-p)
        # a0 = (1-T) * p/(1-p), b0 = a0 * Y
        a0 = (1.0 - T) * p / (1.0 - p)
        b0 = a0 * Y
        A0 = a0.sum()
        B0 = b0.sum()
        mu0 = B0 / A0

        denom1 = A1 / n
        denom0 = A0 / n
        phi1 = (b1 - mu1 * a1) / denom1
        phi0 = (b0 - mu0 * a0) / denom0

        phi = phi1 - phi0

    # -------- mu1: weighted treated mean (IPW Hájek ratio) --------
    elif estimand == "mu1":
        # Here we mimic a generic Hájek ratio m = B/A:
        #   a_i = T_i * w_i
        #   b_i = T_i * w_i * Y_i
        # with IPW weights w_i based on p.
        #
        # Use stabilized weights with frequency weight w* = 1:
        #   d_i = T_i p_i + (1-T_i)(1-p_i)
        #   w_i = 1 / d_i
        d = T * p + (1.0 - T) * (1.0 - p)
        w = 1.0 / d

        a = T * w
        b = T * w * Y
        A = a.sum()
        B = b.sum()
        muY = B / A

        denom = A / n
        phi = (b - muY * a) / denom

    # -------- mu0: weighted control mean (IPW Hájek ratio) --------
    elif estimand == "mu0":
        # Hájek ratio: m = B / A
        # a_i = (1 - T_i) * w_i
        # b_i = (1 - T_i) * w_i * Y_i
        # with IPW weights w_i = 1 / d_i

        d = T * p + (1.0 - T) * (1.0 - p)
        w = 1.0 / d

        a = (1.0 - T) * w
        b = (1.0 - T) * w * Y
        A = a.sum()
        B = b.sum()
        muY = B / A

        denom = A / n
        phi = (b - muY * a) / denom

    else:
        raise ValueError(f"Unknown estimand '{estimand}'. use 'ATE', 'ATT', 'mu0' or 'mu1'.")

    return phi

def lbc_net_moments(propensity_scores, treatment, Z, ck, h, ate=1, kernel_id=0, balance_lambda =1.0):
    """
    Compute the per-observation influence-function contributions of the
    LBC-Net moment conditions (local balance + calibration).

    This function produces φ_i(θ) = ∂Q/∂p_i for each observation,
    where Q is the unified LBC-Net loss:

        Q = E_k[ ||B_k||^2 + C_k^2 ]

    with:
      - B_k = local balance moment at kernel center c_k
      - C_k = calibration moment at c_k

    These per-observation gradients are needed to construct:
      - the plug-in term for influence functions,
      - the Jacobian-vector products required by if_var(),
      - the Hessian–vector implicit products.

    Parameters
    ----------
    propensity_scores : torch.Tensor, shape (N,)
        Estimated propensities p_i from the trained LBC-Net.
    treatment : torch.Tensor, shape (N,)
        Treatment assignment T_i ∈ {0,1}.
    Z : torch.Tensor, shape (N, p)
        Covariate matrix, typically Z_norm (with intercept).
    ck : torch.Tensor, shape (K,)
        Kernel center grid c_k ∈ (0,1).
    h : torch.Tensor, shape (K,)
        Bandwidths h_k for kernel smoothing.
    ate : {0,1}, default 1
        1 for ATE target, 0 for ATT target (affects moments).
    kernel_id : {0,1,2}, default 0
        Kernel type:
           0 = Gaussian
           1 = Uniform
           2 = Epanechnikov
    balance_lambda : float, default 1.0
        Scaling factor for calibration moments in the loss.

    Returns
    -------
    phi_i : torch.Tensor, shape (N, K*(p+1))
        For each observation i, concatenates:
          - local balance components for each kernel center (K*p entries)
          - local calibration components for each center (K entries)
        i.e., total K*(p+1) moment contributions.
    """
    tiny = 1e-6

    # Kernel weights ω(c_k, p_i)
    kernel_w = omega_calculate(propensity_scores, ck, h, kernel_id)
    kernel_w = torch.where(
        (torch.abs(kernel_w) < tiny) & (kernel_w != 0),
        torch.full_like(kernel_w, tiny),
        kernel_w,
    )

    N, p = Z.shape
    K = len(ck)

    # w*(p) = 1 (ATE) or w*(p) = p (ATT)
    w_star = torch.ones_like(propensity_scores) if ate == 1 else propensity_scores
    # Shape [N, K]
    w = kernel_w * w_star.unsqueeze(1)

    # d_i = T_i p_i + (1-T_i)(1-p_i)
    d = treatment * propensity_scores + (1 - treatment) * (1 - propensity_scores)

    # Local balance score:
    #   V_i = ((2T_i - 1)/d_i) Z_i
    V = ((2 * treatment - 1) / d).unsqueeze(1) * Z         # [N, p]

    # φ_B (local balance contributions), shape [N,K,p]
    phiB = w.unsqueeze(2) * V.unsqueeze(1)

    # φ_C (local calibration contributions), shape [N,K]
    phiC = (w * (treatment - propensity_scores).unsqueeze(1)) / (ck * (1 - ck))
    phiC_scaled = balance_lambda * phiC

    # Flatten: concatenate (K*p) + K = K*(p+1) components
    phi_i = torch.cat([phiB.reshape(N, K * p), phiC_scaled], dim=1)

    return phi_i

def _flatten_grads(grads):
    """
    Flatten a list of parameter gradients into a single 1D vector.

    Parameters
    ----------
    grads : list of torch.Tensor
        Each entry corresponds to ∂L/∂θ_j for one network parameter tensor.

    Returns
    -------
    flat : torch.Tensor, shape (total_params,)
        Concatenation of all gradients reshaped to 1D.
    """
    return torch.cat([g.reshape(-1) for g in grads])

def if_var(
    model,          # propensity NN, outputs p in (0,1)
    T,              # [N] {0,1}
    Y,              # [N] outcome for Δ
    Z,              # [N, p] covariates used in moments (can include intercept)
    ck, h,          # [K], [K] kernel params
    phi_ipw,        # [N] plug-in IPW IF for chosen estimand
    ate=1, 
    estimand="ATE", # {"ATE","ATT","Y"} – affects moments and Δ
    kernel_id=0,    # pass-through for omega_calculate
    balance_lambda=1.0,
):
    """
    Influence-function-based SE for an IPW estimand with LBC-Net PS.

    This function computes the chain-rule correction term for the EIF of a
    target estimand (ATE / ATT / Y), accounting for the fact that the
    propensity scores p(·; θ) are estimated by the LBC-Net neural network.

    Conceptually, it builds:
        - ψ_i(θ): stacked LBC-Net moment conditions (local balance + calibration)
        - M  = ∂ m̄(θ)/∂θ, where m̄(θ) = E_n[ψ_i(θ)]
        - g  = ∂ Δ(θ)/∂θ, where Δ is the IPW estimand
        - chain term per obs:  -ψ_i M (M^T M)^{-1} g

    Then it combines this chain term with the plug-in IF φ_ipw (treating p fixed)
    to obtain the total influence function:
        φ_i = φ_ipw,i + chain_i

    Finally, it returns an SE based on Var(φ_i)/n.

    Parameters
    ----------
    model : torch.nn.Module
        Trained LBC-Net model mapping Z -> propensity p in (0,1).
    T : torch.Tensor, shape (N,)
        Treatment assignment (0/1).
    Y : torch.Tensor, shape (N,)
        Outcome used in the estimand Δ.
    Z : torch.Tensor, shape (N, p)
        Covariate matrix (often Z_norm, with intercept).
    ck : torch.Tensor, shape (K,)
        Kernel centers.
    h : torch.Tensor, shape (K,)
        Kernel bandwidths.
    phi_ipw : torch.Tensor, shape (N,)
        Plug-in IF for the chosen estimand, treating p as fixed
        (from plug_in_if(..., estimand=...)).
    ate : {0,1}, default 1
        1 for ATE target, 0 for ATT target (affects moments).
    estimand : {"ATE","ATT","mu1", "mu0"}, default "ATE"
        Target estimand for the EIF.
    kernel_id : {0,1,2}, default 0
        Kernel type: 0=Gaussian, 1=Uniform, 2=Epanechnikov.

    Returns
    -------
    se : torch.Tensor (scalar)
        Estimated standard error of the IPW estimand.
    """

    # --- forward pass for propensity, no graph needed here ---
    p = model(Z).squeeze()           # [N], already sigmoid+clipped in your net
    params = tuple(model.parameters())

    # --- build per-observation stacked moments ψ_i and sample mean m̄ ---
    # First: ψ_i using detached p, just to get numeric values and scaling.
    psi_i = lbc_net_moments(
        propensity_scores=p.detach(),
        treatment=T,
        Z=Z,
        ck=ck,
        h=h,
        ate=ate,
        kernel_id=kernel_id,
        balance_lambda=balance_lambda
    )                                # [N, q]
    N, q = psi_i.shape

    # Now, rebuild ψ_i with graph-enabled p(θ) so that we can differentiate.
    p_graph = model(Z).squeeze()     # [N], graph-enabled
    psi_i_graph = lbc_net_moments(
        propensity_scores=p_graph,
        treatment=T,
        Z=Z,
        ck=ck,
        h=h,
        ate=ate,
        kernel_id=kernel_id,
        balance_lambda=balance_lambda
    )                                # [N, q] with graph
    mbar_graph = psi_i_graph.mean(0) # [q]

    # --- exact Jacobian M = ∂m̄/∂θ, stacked row by row ---
    rows = []
    for j in range(q):
        grads_j = torch.autograd.grad(
            outputs=mbar_graph[j],
            inputs=params,
            retain_graph=True,
            allow_unused=False,
        )
        rows.append(_flatten_grads(grads_j))
    M = torch.stack(rows, dim=0)     # [q, dθ]

    # --- Stabilize by scaling columns of ψ and corresponding rows of M ---
    with torch.no_grad():
        col_scale = psi_i.std(dim=0) + 1e-8  # [q]

    psi_s = psi_i / col_scale                # [N, q]
    M_s   = M / col_scale.unsqueeze(1)       # [q, dθ]

    # --- exact gradient g = ∂Δ/∂θ for chosen estimand ---
    Delta = ipw_est(Y, T, p_graph, estimand=estimand)  # scalar
    g_params = torch.autograd.grad(
        outputs=Delta,
        inputs=params,
        retain_graph=False,
        allow_unused=False,
    )
    gvec = _flatten_grads(g_params)          # [dθ]

    # --- chain correction: -ψ_i M (M^T M)^{-1} g using SVD-based ridge ---
    U, S, Vh = torch.linalg.svd(M_s, full_matrices=False)  # M_s = U Σ V^T
    tau = 1e-3 * S.max()                                   # spectral floor
    mask = (S >= tau)
    S_kept = S[mask]
    V_kept = Vh[mask].T
    g_proj = (Vh @ gvec)[mask]

    lambda_adaptive = 0.01 * (S_kept**2).mean()

    # Ridge in singular space: b = V ( (V^T g) / (Σ^2 + λ) )
    b = V_kept @ (g_proj / (S_kept**2 + lambda_adaptive))  # [dθ]

    # S_i = ψ_i M (M^T M + λ I)^{-1} g  (implemented via scaled version)
    S_chain = psi_s @ M_s @ b                              # [N]
    chain_per_obs = -S_chain                               # [N]

    # --- total IF, variance, and SE ---
    phi = chain_per_obs + phi_ipw                          # [N]
    phi_centered = phi - phi.mean()
    var = (phi_centered.pow(2).sum() / (N - 1)) / N        # Var(φ)/n
    se = torch.sqrt(var)                                   # scalar tensor

    return se

def ipw_surv_na(time, delta, Tr, ps, t_grid):
    """
    IPW Nelson–Aalen survival estimator and survival difference on a time grid.

    This computes IPW-weighted Nelson–Aalen cumulative hazards separately for
    A = 1 and A = 0, then transforms to survival curves S1(t), S0(t) and their
    difference Δ(t) = S1(t) - S0(t) at each t in t_grid.

    Parameters
    ----------
    time : torch.Tensor, shape [n]
        Event or censoring times.
    delta : torch.Tensor, shape [n]
        Event indicator (1 = event, 0 = censored).
    Tr : torch.Tensor, shape [n]
        Treatment indicator (0/1).
    ps : torch.Tensor, shape [n]
        Estimated propensity scores ê(X). MUST keep grad for autograd.
    t_grid : torch.Tensor or array-like, shape [G] or scalar
        Time points at which to evaluate S1, S0, and Δ.
        Assumed (and recommended) to be sorted in ascending order.

    Returns
    -------
    S1_hat : torch.Tensor, shape [G]
        Estimated survival function under treatment at each t in t_grid.
    S0_hat : torch.Tensor, shape [G]
        Estimated survival function under control at each t in t_grid.
    diff_hat : torch.Tensor, shape [G]
        Estimated survival difference S1_hat - S0_hat at each t in t_grid.

    Notes
    -----
    - Returns TENSORS so that autograd can propagate gradients through ps.
    - If there are no events at or before max(t_grid), we return S1 = S0 = 1
      (vector of ones) and diff = 0 (vector of zeros).
    """

    # Keep computation graph for ps; just align dtypes/devices
    dtype = ps.dtype
    device = ps.device

    time  = time.to(device=device, dtype=dtype)
    delta = delta.to(device=device, dtype=dtype)
    Tr    = Tr.to(device=device, dtype=dtype)
    # ps already on correct device/dtype with grad

    # t_grid: convert to 1D tensor on same device
    if not torch.is_tensor(t_grid):
        t_grid = torch.tensor(t_grid, dtype=dtype, device=device)
    t_grid = t_grid.to(device=device, dtype=dtype).view(-1)  # [G]
    G = t_grid.shape[0]

    # Weights for treated and control arms
    # (ω*(p) = 1; any ate flag is ignored here for now)
    w1 = Tr / ps                  # A = 1
    w0 = (1.0 - Tr) / (1.0 - ps)  # A = 0

    # Distinct event times (any arm) up to max(t_grid)
    t_max = torch.max(t_grid)
    mask_event = (delta == 1.0) & (time <= t_max)
    event_times = torch.unique(time[mask_event])
    event_times, _ = torch.sort(event_times)

    # If no events ≤ max(t_grid), survival ≈ 1 at all grid times
    if event_times.numel() == 0:
        S1_hat = torch.ones(G, dtype=dtype, device=device)
        S0_hat = torch.ones(G, dtype=dtype, device=device)
        diff_hat = S1_hat - S0_hat  # zeros
        return S1_hat, S0_hat, diff_hat

    # Build at-risk and jump indicators over event time grid
    # event_times: [J], time: [n]
    t_mat = event_times.unsqueeze(0)       # [1, J]
    time_mat = time.unsqueeze(1)           # [n, 1]

    Y  = (time_mat >= t_mat).to(dtype)     # [n, J], at risk
    dN = ((time_mat == t_mat) &
          (delta.unsqueeze(1) == 1.0)).to(dtype)  # [n, J], event at t_j

    def arm_surv(w_a):
        """
        Compute survival S_a(t) on t_grid for one arm a using IPW Nelson–Aalen.
        """
        WA = w_a.unsqueeze(1)   # [n, 1]

        num = (WA * dN).sum(dim=0)         # [J]
        den = (WA * Y).sum(dim=0)         # [J]
        den = den.clamp_min(1e-10)
        dLambda_hat = num / den           # [J]

        # Cumulative hazard at each event time: Λ(t_j) = sum_{k ≤ j} dΛ_k
        # We don't actually need Λ at each event; we can integrate via masks
        # for each t_grid: Λ(t) = sum_{j: t_j ≤ t} dΛ_j
        # t_grid: [G], event_times: [J]
        mask_t = (t_grid.unsqueeze(1) >= event_times.unsqueeze(0)).to(dtype)  # [G, J]
        Lambda_grid = (mask_t * dLambda_hat.unsqueeze(0)).sum(dim=1)          # [G]

        S_hat = torch.exp(-Lambda_grid)    # [G]
        return S_hat

    # Arm-specific survival curves (TENSORS of shape [G])
    S1_hat = arm_surv(w1)
    S0_hat = arm_surv(w0)

    # Survival difference (TENSOR [G])
    diff_hat = S1_hat - S0_hat

    return S1_hat, S0_hat, diff_hat

def plugin_if_surv(time, delta, Tr, ps, t_grid):
    """
    Plug-in influence function for the survival difference
        Δ(t) = S1(t) - S0(t)
    based on IPW Nelson–Aalen estimator, treating propensity scores as fixed.

    This computes the plug-in IF at a grid of time points t_grid, returning
    an n x G tensor, where G = len(t_grid).

    Parameters
    ----------
    time   : 1D tensor, shape [n]
        Observed times T_i (event or censoring).
    delta  : 1D tensor, shape [n]
        Event indicator (1 = event, 0 = censored).
    Tr     : 1D tensor, shape [n]
        Treatment indicator A_i (0/1).
    ps     : 1D tensor, shape [n]
        Propensity scores e(X_i) = P(A=1 | X_i).
    t_grid : 1D tensor or array-like, shape [G] or scalar
        Time points at which to evaluate the survival difference.
        Assumed (and recommended) to be sorted in ascending order.
    ate    : int, default 1
        Placeholder for estimand flag; here we always use ATE-type weights
        (ω*(p) = 1), so this argument is currently ignored.

    Returns
    -------
    phi : 2D tensor, shape [n, G]
        Plug-in influence function values for Δ(t_g) at each t_g in t_grid.
        Row i is IF_i(Δ(·)), column g is IF at t_grid[g].
    """

    # Work in double precision, treating ps as FIXED (no autograd)
    time  = time.clone().detach().double()
    delta = delta.clone().detach().double()
    Tr    = Tr.clone().detach().double()
    ps    = ps.clone().detach().double()

    # t_grid -> 1D double tensor
    if not torch.is_tensor(t_grid):
        t_grid = torch.tensor(t_grid, dtype=torch.double)
    t_grid = t_grid.clone().detach().double().view(-1)   # [G]
    G = t_grid.shape[0]

    n = time.shape[0]

    # ATE-type weights for treated and control arms (ω*(p) = 1)
    w1 = Tr / ps                   # A = 1
    w0 = (1.0 - Tr) / (1.0 - ps)   # A = 0

    # Distinct event times (any arm) up to max(t_grid)
    t_max = t_grid.max()
    mask_event = (delta == 1.0) & (time <= t_max)
    event_times = torch.unique(time[mask_event])
    event_times, _ = torch.sort(event_times)

    # If no events ≤ max(t_grid), survival ≈ 1 and IF ≈ 0
    if event_times.numel() == 0:
        return torch.zeros(n, G, dtype=torch.double)

    # Build at-risk and jump indicators over the event time grid
    # event_times: [J]; time: [n]
    t_mat = event_times.unsqueeze(0)   # [1, J]
    time_mat = time.unsqueeze(1)       # [n, 1]

    Y  = (time_mat >= t_mat).double()  # [n, J], at risk
    dN = ((time_mat == t_mat) &
          (delta.unsqueeze(1) == 1.0)).double()  # [n, J], event at t_j

    def arm_if(w_a):
        """
        Compute IF for S_a(t) on t_grid for one arm a using plug-in formulas.

        Returns
        -------
        IF_S : [n, G] tensor
            Influence function for S_a(t_g) at each grid time t_g.
        S_hat : [G] tensor
            Estimated survival S_a(t_g) at each t_g.
        """
        WA = w_a.unsqueeze(1)   # [n, 1]

        # Ψ2_hat(t_j; a) = E_n[w_a Y(t_j)]  (empirical mean over i)
        Psi2_hat = (WA * Y).mean(dim=0)      # [J]
        Psi2_hat = Psi2_hat.clamp_min(1e-10)

        # Weighted NA increments:
        # dΛ_hat_a(t_j) = sum_i w_a,i dN_i(t_j) / sum_i w_a,i Y_i(t_j)
        num = (WA * dN).sum(dim=0)           # [J]
        den = (WA * Y).sum(dim=0)           # [J]
        den = den.clamp_min(1e-10)
        dLambda_hat = num / den             # [J]

        # Influence function for Λ_a(t_g):
        # IF_i(Λ_a(t_g)) = Σ_{j: t_j ≤ t_g} 1/Ψ2_hat(t_j) * w_a,i {dN_ij - Y_ij dΛ_hat(t_j)}
        inv_Psi2 = 1.0 / Psi2_hat           # [J]
        # term_ij = 1/Ψ2_j * w_i * (dN_ij - Y_ij dΛ_hat_j)
        term = inv_Psi2.unsqueeze(0) * WA * (dN - Y * dLambda_hat.unsqueeze(0))  # [n, J]

        # For each grid time t_g, sum over j with t_j ≤ t_g
        # mask_t[g, j] = 1{t_j ≤ t_g}
        mask_t = (t_grid.unsqueeze(1) >= event_times.unsqueeze(0)).double()  # [G, J]
        # IF_lambda[i, g] = Σ_j mask_t[g, j] * term[i, j]
        IF_lambda = torch.matmul(term, mask_t.t())   # [n, G]

        # Cumulative hazard at each t_g:
        # Λ_hat(t_g) = Σ_{j: t_j ≤ t_g} dΛ_hat(t_j)
        Lambda_grid = torch.matmul(mask_t, dLambda_hat)  # [G]

        # Survival at each grid time: S_hat_a(t_g) = exp(-Λ_hat(t_g))
        S_hat = torch.exp(-Lambda_grid)  # [G]

        # IF for survival: IF_i(S_a(t_g)) = -S_a(t_g) * IF_i(Λ_a(t_g))
        IF_S = -IF_lambda * S_hat.unsqueeze(0)  # [n, G]

        return IF_S, S_hat

    # Arm-specific IFs and survival estimates
    IF_S1, S1_hat = arm_if(w1)   # [n, G], [G]
    IF_S0, S0_hat = arm_if(w0)   # [n, G], [G]

    # IF for survival difference Δ(t_g) = S1(t_g) - S0(t_g)
    phi = IF_S1 - IF_S0   # [n, G]

    return phi

def if_var_surv(
    ps_model,      # propensity NN, outputs ps in (0,1)
    A,            # [N] {0,1} treatment
    time,         # [N] survival time
    delta,        # [N] event indicator
    Z,            # [N, p] covariates (normalized, with intercept) used in LBC-Net
    ck, h,        # [K], [K] kernel params
    phi_ipw,      # [N, G] plug-in IF for Δ(t) treating PS as fixed
    t_grid,       # [G] time grid
    ate=1,
    kernel_id=0
):
    """
    Influence-function-based variance for the survival difference Δ(t) over a grid.

    This function computes, for each time t_g in t_grid, the IF-corrected
    variance of Δ(t_g) = S1(t_g) - S0(t_g), accounting for the effect of
    the LBC-Net propensity model via implicit differentiation.

    Parameters
    ----------
    ps_model : torch.nn.Module
        Trained LBC-Net propensity model. Forward pass: ps_model(Z) -> ê(X).
    A : 1D tensor, shape [N]
        Treatment indicator (0/1).
    time : 1D tensor, shape [N]
        Survival times.
    delta : 1D tensor, shape [N]
        Event indicator (1 = event, 0 = censored).
    Z : 2D tensor, shape [N, p]
        Covariate matrix used for LBC-Net (already normalized, with intercept).
    ck : 1D tensor, shape [K]
        Kernel centers.
    h : 1D tensor, shape [K]
        Bandwidths corresponding to ck.
    phi_ipw : 2D tensor, shape [N, G]
        Plug-in influence function for Δ(t) at each grid time, treating PS as fixed
        (from `plugin_if_surv`).
    t_grid : 1D tensor, shape [G]
        Time grid at which survival differences are evaluated.
    ate : int, default 1
        Estimand flag; here we always use ATE-type weights (ω*(p) = 1).
    kernel_id : int, default 0
        Kernel identifier (0 = Gaussian, 1 = Uniform, 2 = Epanechnikov).

    Returns
    -------
    var_diff : 1D tensor, shape [G]
        IF-based variance estimates of Δ(t_g) at each t_g in t_grid.
    """

    # Ensure proper dtypes/devices for autograd
    device = next(ps_model.parameters()).device
    A      = A.to(device=device, dtype=torch.float32)
    time   = time.to(device=device, dtype=torch.float32)
    delta  = delta.to(device=device, dtype=torch.float32)
    Z      = Z.to(device=device, dtype=torch.float32)
    ck     = ck.to(device=device, dtype=torch.float32)
    h      = h.to(device=device, dtype=torch.float32)

    if not torch.is_tensor(t_grid):
        t_grid = torch.tensor(t_grid, dtype=torch.float32, device=device)
    t_grid = t_grid.to(device=device, dtype=torch.float32).view(-1)  # [G]
    G = t_grid.shape[0]

    phi_ipw = phi_ipw.to(device=device, dtype=torch.float32)  # [N, G]

    # --- forward pass for propensity (once, detached version for ψ_i) ---
    with torch.no_grad():
        p_detached = ps_model(Z).squeeze()  # [N], numeric ps

    # --- build per-observation stacked moments ψ_i (no graph needed) ---
    psi_i = lbc_net_moments(
        propensity_scores=p_detached,
        treatment=A,
        Z=Z,
        ck=ck,
        h=h,
        ate=ate,
        kernel_id=kernel_id,
    )  # [N, q]
    N, q = psi_i.shape

    # --- Jacobian M = ∂ mbar / ∂θ via autograd (only once, shared for all times) ---
    params = tuple(ps_model.parameters())

    p_graph = ps_model(Z).squeeze()  # [N], graph-enabled
    psi_i_graph = lbc_net_moments(
        propensity_scores=p_graph,
        treatment=A,
        Z=Z,
        ck=ck,
        h=h,
        ate=ate,
        kernel_id=kernel_id,
    )  # [N, q], graph-enabled
    mbar_graph = psi_i_graph.mean(0)  # [q]

    rows = []
    for j in range(q):
        grads_j = torch.autograd.grad(
            outputs=mbar_graph[j],
            inputs=params,
            retain_graph=True,
            allow_unused=False,
        )
        rows.append(_flatten_grads(grads_j))  # helper: concat per-param grads
    M = torch.stack(rows, dim=0)  # [q, dθ]

    # --- column scaling for numerical stability ---
    with torch.no_grad():
        col_scale = psi_i.std(dim=0) + 1e-8  # [q]

    psi_s = psi_i / col_scale          # (N x q)
    M_s   = M / col_scale.unsqueeze(1) # (q x dθ)

    # --- SVD for (M^T M)^(-1) via spectral shrinkage (once) ---
    U, S, Vh = torch.linalg.svd(M_s, full_matrices=False)  # M_s = U Σ V^T
    tau = 1e-3 * S.max()
    mask = (S >= tau)
    S_kept = S[mask]                      # [r]
    V_kept = Vh[mask].T                   # (dθ x r)
    lambda_adaptive = 0.01 * (S_kept**2).mean()

    # --- compute Δ(t) for all grid times once (graph-enabled) ---
    S1_hat_all, S0_hat_all, Delta_hat_all = ipw_surv_na(
        time,
        delta,
        A,
        p_graph,
        t_grid,
    )  # each is [G]

    # --- allocate variance vector for Δ(t_g) ---
    var_list = []

    # Loop over each grid time to get g(t_g), chain term, and variance
    for g_idx in range(G):
        # Gradient of Δ(t_g) w.r.t. θ
        Delta_g = Delta_hat_all[g_idx]  # scalar
        grads_g = torch.autograd.grad(
            outputs=Delta_g,
            inputs=params,
            retain_graph=True,
            allow_unused=False,
        )
        gvec = _flatten_grads(grads_g)  # [dθ]

        # project gradient into singular space of M_s
        g_proj = (Vh @ gvec)[mask]  # [r]

        # ridge solution: b = V_kept (g_proj / (Σ^2 + λ))
        b = V_kept @ (g_proj / (S_kept**2 + lambda_adaptive))  # [dθ]

        # chain term per observation: - ψ_i M b (using scaled ψ & M for stability)
        S_chain = psi_s @ M_s @ b          # [N]
        chain_per_obs = -S_chain           # [N]

        # total IF at t_g = plug-in IF + chain correction
        phi_g = chain_per_obs + phi_ipw[:, g_idx]  # [N]

        # variance from centered IF
        phi_centered = phi_g - phi_g.mean()
        var_g = (phi_centered.pow(2).sum() / (N - 1)) / (N**2)  # scalar
        var_list.append(var_g)

    var_diff = torch.stack(var_list)  # [G]
    return var_diff
