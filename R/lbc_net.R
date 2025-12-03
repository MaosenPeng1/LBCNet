#' Estimate Propensity Scores Using LBC-Net
#'
#' @description `lbc_net` estimates propensity scores using the LBC-Net model,
#' a deep learning method designed to enhance covariate balance.
#' It integrates a Variational Autoencoder (VAE) with a customized
#' neural network to estimate treatment probabilities for causal inference.
#' 
#' When an outcome `Y` is supplied, `lbc_net` also computes inverse probability
#' weighted (IPW) estimates of the causal effect (ATE or ATT) and, when enabled,
#' corresponding influence-function–based standard errors and confidence intervals.
#'
#' @param data an optional data frame, list, or environment (or an object
#'   coercible by `as.data.frame` to a data frame) containing the variables
#'   specified in `formula`. If a variable is not found in `data`, it is taken
#'   from `environment(formula)`, typically the environment from which
#'   `lbc_net()` is called. If `formula` is not provided, `Z` (a matrix of
#'   covariates) and `Tr` (a numeric treatment assignment vector) must be
#'   explicitly supplied.
#'
#' @param formula An object of class `"formula"` (or one that can be coerced to
#'   that class), specifying a symbolic description of the model in the form
#'   `Tr ~ X1 + X2 + ...`. If `formula` is provided, `Z` and `Tr` are extracted
#'   from `data`. If omitted, `Z` and `Tr` must be supplied explicitly.
#'
#' @param Z A numeric matrix or data frame of covariates. Required if `formula` is not provided.
#'   Each row represents an observation, and each column represents a covariate.
#'   If `formula` is used, `Z` is extracted from `data` automatically.
#'
#' @param Tr A numeric vector representing treatment assignment (typically 0/1).
#'   Required if `formula` is not provided. Must have the same number of rows
#'   as `Z`. If `formula` is used, `Tr` is extracted from `data` automatically.
#'   
#' @param Y Optional numeric vector of observed outcomes. If provided, `lbc_net`
#'   will, in addition to estimating propensity scores, compute inverse probability
#'   weighted estimates of a causal estimand (e.g., ATE or ATT) using the fitted
#'   LBC-Net model. `Y` can be continuous or binary; the IPW formula is the same,
#'   only the interpretation differs.
#'
#' @param estimand Character string specifying the target estimand when an outcome
#'   `Y` is supplied. Available options are:
#'   \describe{
#'     \item{`"ATE"`}{Average Treatment Effect. The frequency weight function
#'       \eqn{\omega^{*}(p_i) = 1} targets the combined population.}
#'     \item{`"ATT"`}{Average Treatment Effect on the Treated. The frequency weight
#'       function \eqn{\omega^{*}(p_i) = p_i} upweights units that are likely to be treated.}
#'     \item{`"Y"`}{Weighted mean outcome among the treated group, using IPW weights
#'       derived from the fitted propensity scores.}
#'   }
#'   If `Y` is `NULL`, `estimand` is ignored and `lbc_net` only fits the propensity model. 
#'   See **Details** for more information on ATT, ATE, and their corresponding weighting schemes.
#'
#' @param K an integer specifying the number of grid center points used to
#'   compute kernel weights in the local neighborhood. These weights are
#'   used to assess balance and calibration conditions. If specified, `ck`
#'   is automatically computed as `k / (K + 1)`, where `k = 1, ..., K`.
#'   The default is `99`, generating grid points from `0.01` to `0.99`.
#'   See **Details** for more information on kernel weights.
#'
#' @param rho a numeric value specifying the span used to determine the adaptive
#'   bandwidth `h` when `h` is not provided. The span controls the proportion of
#'   data included in the local neighborhood, ensuring a sufficient sample size
#'   for accurate training. The choice of `rho` influences local balance assessment
#'   and should be selected based on the data structure. While cross-validation
#'   can be used to approximate an optimal span, user discretion is advised.
#'   The default is `0.15`.
#'
#' @param na.action A function to specify the action to be taken if NAs are found.
#'   The default action is for the procedure to fail. An alternative is `na.omit`,
#'   which leads to rejection of cases with missing values on any required variable.
#'   (NOTE: If given, this argument must be named.)
#'
#' @param gpu An integer specifying the GPU device ID for computation if CUDA is available.
#'   If set to `0`, the function will attempt to use the default GPU. If CUDA is unavailable,
#'   computations will automatically fall back to the CPU.
#'
#' @param show_progress A logical value indicating whether to display a progress bar during training.
#'   If `TRUE` (default), displays elapsed time, remaining time, loss values, and training speed
#'   (iterations per second). This helps monitor training progress efficiently. Set to `FALSE` to disable the display.
#'
#' @param ... Additional parameters for model tuning, including:
#'   \describe{
#'     \item{`ck`}{a numeric vector of kernel center points. Values should be
#'   strictly between `0` and `1`. If `NULL`, `ck` is automatically computed
#'   using the default `K`. If provided, user-defined grid points should
#'   adhere to the constraint `0 < ck < 1`.}
#'
#'     \item{`h`}{A numeric vector of bandwidth values for kernel weighting. By
#'   default, an adaptive bandwidth is automatically computed via preliminary
#'   probabilities estimated using logistic regression (`glm`), given `ck`
#'   and the span parameter `rho`. If `NULL`, `h` is computed using
#'   \code{\link{span_bw}}.}
#'
#'     \item{`kernel`}{A character string specifying the kernel function used for local
#'   weighting. The default is `"gaussian"`. Available options include:
#'      \itemize{
#'        \item `"gaussian"` (default): Ensures smooth weighting and continuity.
#'        \item `"uniform"`: Assigns equal weight to all observations within a fixed bandwidth.
#'        \item `"epanechnikov"`: Gives higher weight to observations closer to the kernel center.
#'      }
#'   See **Details** for more information on kernel weighting and its role in local
#'   balance estimation.}
#'
#'     \item{`seed`}{An integer specifying the random seed for reproducibility in training the neural network.
#'   This seed is applied to PyTorch using `torch.manual_seed(seed)`, ensuring consistent results
#'   across runs when using stochastic optimization methods.}
#'
#'     \item{`hidden_dim`}{An integer specifying the number of hidden units in the LBC-Net model.
#'   A rule of thumb is to set this as two to three times the number of covariates,
#'   but it should be significantly smaller than the sample size to prevent overfitting.
#'   Default is `100`.}
#'
#'     \item{`num_hidden_layers`}{An integer specifying the number of hidden layers in the LBC-Net model.
#'   Default is `1`, which results in a three-layer network overall (input layer, one hidden layer, and output layer).}
#'
#'     \item{`vae_epochs`}{An integer specifying the number of training epochs for the Variational Autoencoder (VAE).
#'   This determines how long the VAE component of the model is trained before being used in LBC-Net.
#'   Default is `250`.}
#'
#'     \item{`vae_lr`}{A numeric value specifying the learning rate for training the VAE using the Adam optimizer.
#'   Controls how quickly the model updates its parameters during training. Default is `0.01`.}
#'
#'     \item{`max_epochs`}{An integer specifying the maximum number of training epochs for LBC-Net.
#'   Early stopping is applied based on `lsd_threshold` to prevent unnecessary training.
#'   Default is `5000`.}
#'
#'     \item{`lr`}{A numeric value specifying the initial learning rate for LBC-Net training using the Adam optimizer.
#'   The learning rate controls how much the model updates during each training step.
#'   Default is `0.05`.}
#'
#'     \item{`weight_decay`}{A numeric value specifying the regularization parameter in the Adam optimizer for LBC-Net.
#'   Helps prevent overfitting by penalizing large weights in the model.
#'   Default is `1e-5`.}
#'
#'     \item{`balance_lambda`}{A numeric value controlling the relative contributions of the local balance loss (\eqn{Q_1})
#'   and calibration loss (\eqn{Q_2}) in the objective function, where the total loss is defined as
#'   \eqn{Q = Q_1 + \lambda Q_2}. Default is `1.0`. }
#'
#'     \item{`epsilon`}{A small numeric value controlling the lower and upper bounds of the
#'   estimated propensity scores. The default is `0.001`, ensuring scores remain within
#'   \eqn{[\epsilon, 1 - \epsilon]} for numerical stability, particularly in cases of
#'   poor overlap. Setting `epsilon = 0` reverts to the standard logit link function.
#'   See **Details** for more on its role in model stabilization.}
#'
#'     \item{`lsd_threshold`}{A numeric value defining the stopping criterion based on the Local Standardized mean Difference (LSD).
#'   Training stops when the rolling average of the maximum local balance falls below this threshold.
#'   The default `lsd_threshold = 2` balances efficiency and precision. In cases of poor overlap or small
#'   sample sizes, a more relaxed threshold (e.g., `5\%` or `10\%`) may be used to allow more flexibility in training.}
#'
#'     \item{`rolling_window`}{An integer specifying the number of recent epochs used to compute the rolling average of
#'   the maximum local balance. Default is `5`. The early stopping mechanism is triggered when the rolling average
#'   of the maximum LSD over the most recent `rolling_window` epochs falls below `lsd_threshold`. Specifically,
#'   at every 200-epoch step, the maximum local balance is calculated, and a rolling average over the last
#'   `rolling_window` steps is updated. Training halts when this rolling average drops below `lsd_threshold`,
#'   or when the predefined maximum epochs is reached, ensuring sufficient learning capacity.}
#'   }
#'   
#' @param setup_lbcnet_args List. Optional arguments passed to \code{\link{setup_lbcnet}} for configuring the Python environment.
#'   If Python is not set up, `setup_lbcnet()` is automatically called using these parameters.
#'   Default is `list(envname = "r-lbcnet", create_if_missing = TRUE)`, meaning it will attempt to use the virtual environment `"r-lbcnet"`
#'   and create it if missing.
#'
#' @return An object of class `"lbc_net"`, containing:
#' \itemize{
#'         \item `fitted.values`: Estimated propensity scores.
#'         \item `weights`: Inverse probability weights (IPW).
#'       }
#'
#' Other model components (e.g., `losses`, `parameters`, `Z`, `Tr`) are accessible via `$`
#' or the recommended \code{\link[=getLBC.lbc_net]{getLBC}} function. While direct access (e.g., `fit$fitted.values`)
#' is possible, using `getLBC(fit, "fitted.values")` is recommended for stability and future-proofing.
#'
#' @seealso \code{\link{lbc_net-class}}
#'
#' @details
#' This function optimizes the objective function \eqn{Q(\theta)} using a
#' feed-forward neural network with batch normalization after each layer.
#' Rectified Linear Unit (ReLU) activations are used in hidden layers, while
#' the output layer employs a modified sigmoid activation to ensure propensity
#' scores remain bounded between \eqn{\epsilon} and \eqn{1-\epsilon}:
#'
#' \deqn{
#' p(\mathbf{Z}) = \epsilon + (1 - 2\epsilon)
#' \frac{\exp(S(\mathbf{Z}; \theta))}{1 + \exp(S(\mathbf{Z}; \theta))}
#' }
#'
#' In well-overlapping distributions, \eqn{\epsilon = 0} (logit link function)
#' is effective, while for poor overlap, \eqn{\epsilon = 0.001} stabilizes computation
#' by preventing extreme probabilities (0 or 1). The default \eqn{\epsilon = 0.001}
#' works well in most cases.
#'
#' If categorical covariates with more than two levels are included in `formula` or `Z`,
#' users must manually convert them into dummy (one-hot encoded) variables before fitting the model.
#' This can be done as follows:
#'
#' \preformatted{
#' data$cate <- factor(data$cate)
#' dummy_vars <- model.matrix(~ cate - 1, data = data)
#' data <- cbind(data, dummy_vars)
#' data$cate <- NULL
#' }
#'
#' \strong{Optimization Process}:
#' The model is trained using Adaptive Moment Estimation (ADAM) optimization.
#' Given the complexity of the objective function, a pre-training phase
#' using a Variational Autoencoder (VAE) is incorporated to enhance
#' feature representation. After pre-training, the encoder weights are transferred
#' to initialize the LBC-Net. This initialization improves training stability
#' and propensity score estimation.
#'
#' \strong{Kernel Weighting & Local Inverse Probability Weights (IPW)}:
#' To weigh observations in local neighborhoods, we use kernel smoothing to
#' define local balance weights. The default choice in this package is the
#' Gaussian kernel due to its smoothness and continuity, but users may
#' specify alternative kernels.
#'
#' The general form of the kernel weighting function is:
#' \deqn{\omega(c_k, x) = h_k^{-1} K\left( \frac{c_k - x}{h_k} \right)}
#' where \eqn{h_k} is the location-specific bandwidth, and \eqn{K(x)}
#' is the kernel function. The default Gaussian kernel is given by:
#' \deqn{K(x) = (2\pi)^{-1/2} \exp(-x^2/2)}
#'
#' Alternative Kernel Choices:
#' Users can modify the kernel function for different weighting schemes:
#' \itemize{
#'         \item Uniform Kernel:
#'          \deqn{K(x) = 0.5 \times \mathbf{1}(|x| \leq 1)}
#'         \item Epanechnikov Kernel:
#'          \deqn{K(x) = 0.75 (1 - x^2) \mathbf{1}(|x| \leq 1)}
#'       }
#'
#' The Local Inverse Probability Weight (LIPW) at each local grid center \eqn{c_k} is:
#' \deqn{
#' W_k(p_i) = \frac{ \omega(c_k, p_i)\omega^*(p_i) }{ T_i p_i + (1 - T_i)(1 - p_i) }, \quad i = 1, 2, ..., N.
#' }
#'
#' The frequency weight function \eqn{\omega^{*}(p_i)} determines the target population:
#' setting \eqn{\omega^{*}(p_i) = 1} yields the Average Treatment Effect (ATE), while
#' choosing \eqn{\omega^{*}(p_i) = p_i} results in the Average Treatment Effect on the Treated (ATT).
#'
#' \strong{Training Considerations & Tuning}:
#'
#' - Poor Overlap Situations: If groups have poor overlap
#'   (see \code{\link{mirror_hist}}), achieving the minimum local balance may be difficult.
#'   In such cases, relax `lsd_threshold` and increase `max_epochs`.
#'
#' - Tuning Neural Network Parameters: The local balance (`LSD`) and loss
#'   from \code{\link[=summary.lbc_net]{summary}} can guide tuning. However, the default values
#'   generally work well in most cases.
#'
#' - The LSD metric is used to evaluate local balance and guide hyperparameter tuning.
#'
#' - During training, the model tracks LSD values to determine convergence.
#'
#'   These can be retrieved using: \code{\link{getLBC}}(object, "max_lsd"):
#'     Returns the maximum LSD at last epoch training; \code{\link{getLBC}}(object, "mean_lsd"):
#'     Returns the mean LSD.
#'
#' - For final evaluation:
#'     \itemize{
#'         \item \code{\link{lsd}}(object):
#'          Computes the final LSD values based on the fitted model;
#'         \item \code{\link[=plot.lsd]{plot.lsd}}(object):
#'          Generates visualizations for LSD values across covariates;
#'         \item \code{\link{plot_calib}}(object):
#'          Assesses the local calibration for estimated propensity scores.
#'         \item \code{\link{mirror_hist}}(object):
#'          Mirror histogram for propensity score distribution between groups.
#'       }
#'
#' @examples
#' \dontrun{
#' # Set seed for reproducibility
#' set.seed(123456)
#'
#' # Define sample size
#' n <- 5000
#'
#' # Generate true covariates from a multivariate normal distribution
#' if (requireNamespace("MASS", quietly = TRUE)) {
#'   Z <- MASS::mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
#' }
#'
#' # Generate propensity scores (true model)
#' prop <- 1 / (1 + exp(Z[,1] - 0.5 * Z[,2] +
#'                       0.25 * Z[,3] + 0.1 * Z[,4]))
#'
#' # Assign treatment based on propensity scores
#' Tr <- rbinom(n, 1, prop)
#'
#' # Generate continuous outcome (correctly specified model)
#' Y <- 210 + 27.4 * Z[,1] + 13.7 * Z[,2] +
#'      13.7 * Z[,3] + 13.7 * Z[,4] + rnorm(n)
#'
#' # Estimate propensity scores with a misspecified model
#' X <- cbind(exp(Z[,1] / 2),
#'            Z[,2] * (1 + exp(Z[,1]))^(-1) + 10,
#'            ((Z[,1] * Z[,3]) / 25 + 0.6)^3,
#'            (Z[,2] + Z[,4] + 20)^2)
#'
#' # Combine data into a data frame
#' data <- data.frame(Y, Tr, X)
#' colnames(data) <- c("Y", "Tr", "X1", "X2", "X3", "X4")
#'
#' # --- Fit the LBC-Net Model (propensity only) ---
#'
#' # Option 1: Using formula input (PS + diagnostics only)
#' model_ps <- lbc_net(data = data, formula = Tr ~ X1 + X2 + X3 + X4)
#'
#' # Option 2: Directly using Z and Tr
#' model_ps2 <- lbc_net(Z = X, Tr = Tr)
#'
#' # --- Fit the LBC-Net Model and estimate ATE in one step ---
#' model_ate <- lbc_net(
#'   data     = data,
#'   formula  = Tr ~ X1 + X2 + X3 + X4,
#'   Y        = data$Y,
#'   estimand = "ATE"
#' )
#'
#' print(model_ate)          # basic print
#' summary(model_ate)        # may show effect and PS summaries
#'
#' # Extract effect and SE
#' getLBC(model_ate, "effect")
#' getLBC(model_ate, "se")
#'
#' # --- Performance Evaluation ---
#'
#' # Mirror histogram of propensity scores
#' mirror_hist(model)
#'
#' # Calibration plot
#' plot_calib(model)
#'
#' # Compute and plot the LSD metric
#' lsd.fit <- lsd(model)
#' plot(lsd.fit)
#' }
#'
#' @importFrom stats na.fail model.frame model.response model.matrix glm
#' @importFrom reticulate source_python
#'
#' @export
lbc_net <- function(data = NULL, formula = NULL, Z = NULL, Tr = NULL, Y = NULL,
                    estimand = c("ATE", "ATT", "Y"), K = 99, rho = 0.15, na.action = na.fail,
                    gpu = 0, show_progress = TRUE, ..., setup_lbcnet_args = list()) {
  
  # Ensure Python is properly configured before running setup
  if (!reticulate::py_available(initialize = FALSE)) {
    message("Python environment is not set up. Running `setup_lbcnet()`...")
    do.call(setup_lbcnet, setup_lbcnet_args)  # Automatically configure Python
  } else {
    message("Python is already set up. Skipping `setup_lbcnet()`.")
  }
  
  estimand <- match.arg(estimand)
  ate_flag <- if (estimand %in% c("ATE", "Y")) 1L else 0L

  # Extract additional parameters from ...
  args <- list(...)

  # Extract optional kernel weighting parameters
  ck <- if (!is.null(args$ck)) args$ck else NULL
  h <- if ("h" %in% names(args)) args$h else NULL
  kernel <- if (!is.null(args$kernel)) args$kernel else "gaussian"

  # Extract NN model tuning parameters
  seed <- if (!is.null(args$seed)) args$seed else 100
  hidden_dim <- if (!is.null(args$hidden_dim)) args$hidden_dim else 100
  num_hidden_layers <- if (!is.null(args$num_hidden_layers)) args$num_hidden_layers else 1
  vae_epochs <- if (!is.null(args$vae_epochs)) args$vae_epochs else 250
  vae_lr <- if (!is.null(args$vae_lr)) args$vae_lr else 0.01
  max_epochs <- if (!is.null(args$max_epochs)) args$max_epochs else 5000
  lr <- if (!is.null(args$lr)) args$lr else 0.05
  weight_decay <- if (!is.null(args$weight_decay)) args$weight_decay else 1e-5
  balance_lambda <- if (!is.null(args$balance_lambda)) args$balance_lambda else 1.0
  epsilon <- if (!is.null(args$epsilon)) args$epsilon else 0.001
  lsd_threshold <- if (!is.null(args$lsd_threshold)) args$lsd_threshold else 2
  rolling_window <- if (!is.null(args$rolling_window)) args$rolling_window else 5

  # Load Python script
  script_path <- system.file("python", "lbc_net.py", package = "LBCNet")
  mymodule <- reticulate::import_from_path("lbc_net", path = system.file("python", package = "LBCNet"))

  # Handle formula-based input
  if (!is.null(formula) && !is.null(data)) {
    model_frame <- model.frame(formula, data, na.action = na.action)
    Tr <- model.response(model_frame)
    Z <- model.matrix(attr(model_frame, "terms"), model_frame)

    # Drop intercept if exists
    if ("(Intercept)" %in% colnames(Z)) {
      Z <- Z[, -1, drop = FALSE]
    }
  }

  # Ensure valid input
  if (is.null(Z) || is.null(Tr)) {
    stop("Either provide a formula with a dataframe or specify Z and Tr directly as a matrix and vector.")
  }
  
  if (is.vector(Z) || (is.data.frame(Z) && ncol(Z) == 1)) {
    Z <- matrix(Z, ncol = 1)
  } else if (is.data.frame(Z) || is.matrix(Z)) {
    Z <- as.matrix(Z)
  } else {
    stop("Z must be a numeric vector, matrix, or data frame.")
  }
  
  if (!is.numeric(Tr) || length(Tr) != nrow(Z)) stop("Tr must be a numeric vector.")
  
  if (!is.null(Y)) {
    if (!is.numeric(Y)) stop("Y must be numeric when provided.")
    if (length(Y) != nrow(Z)) {
      stop("Y must have the same length as the number of rows in Z.")
    }
  }

  # Apply NA action
  if (!is.null(Y)) {
    dat_all <- data.frame(Z, Tr = Tr, Y = Y)
    na_result <- na.action(dat_all)
    
    Z  <- as.matrix(na_result[, seq_len(ncol(na_result) - 2), drop = FALSE])
    Tr <- na_result[, ncol(na_result) - 1]
    Y  <- na_result[, ncol(na_result)]
  } else {
    dat_all <- data.frame(Z, Tr = Tr)
    na_result <- na.action(dat_all)
    
    Z  <- as.matrix(na_result[, -ncol(na_result), drop = FALSE])
    Tr <- na_result[, ncol(na_result)]
  }

  # Convert Tr to numeric
  Tr <- as.numeric(Tr)
  
  if (!is.null(Y)) {
    if (all(Tr == 1)) {
      stop("All units are treated (Tr = 1). Cannot compute ATE/ATT.")
    }
    if (all(Tr == 0)) {
      stop("All units are control (Tr = 0). Cannot compute ATE/ATT.")
    }
  }

  # Calculate propensity scores for ck/h calculation
  message("Calculating propensity scores for ck/h calculation...")
  log.fit <- glm(Tr ~ ., data = as.data.frame(Z), family = "binomial")
  ps_log <- log.fit$fitted.values

  # Auto-calculate `ck` based on `K` if not provided
  if (is.null(ck)) {
    ck <- seq(1 / (K + 1), K / (K + 1), length.out = K)
  } else {
    if (!is.numeric(ck) || any(ck <= 0 | ck >= 1)) {
      stop("`ck` must be a numeric vector with values strictly between 0 and 1.")
    }
  }

  # Auto-calculate `h` if not provided
  if (is.null(h)) {
    h <- span_bw(rho, ck, ps_log)
  } else {
    if (!is.numeric(h)) stop("`h` must be a numeric vector.")
    if (length(h) != length(ck)) stop("`h` must have the same length as `ck`.")
  }

  # Ensure Z has column names
  if (is.null(colnames(Z))) {
    colnames(Z) <- paste0("V", seq_len(ncol(Z)))  # Assign generic names V1, V2, V3, ...
  }
  
  if (is.null(Y)) {
    data_df <- as.data.frame(cbind(Tr, Z))
    colnames(data_df)[1] <- "Tr"
    Y_column <- NULL
    compute_variance <- FALSE
  } else {
    data_df <- as.data.frame(cbind(Tr, Z, Y))
    colnames(data_df) <- c("Tr", colnames(Z), "Y")
    Y_column <- "Y"
    compute_variance <- TRUE
  }
  
  # Call Python function and capture results
  result <- mymodule$run_lbc_net(
    data_df = data_df,
    Z_columns = colnames(Z),
    T_column = "Tr",
    Y_column = Y_column,
    estimand = estimand,
    ck = ck,
    h = h,
    kernel = kernel,
    gpu = as.integer(gpu),
    ate = as.integer(ate_flag),
    seed = as.integer(seed),
    hidden_dim = as.integer(hidden_dim),
    L = as.integer(num_hidden_layers+1),
    vae_epochs = as.integer(vae_epochs),
    vae_lr = vae_lr,
    max_epochs = as.integer(max_epochs),
    lr = lr,
    weight_decay = weight_decay,
    balance_lambda = balance_lambda,
    epsilon = epsilon,
    lsd_threshold = lsd_threshold,
    rolling_window = as.integer(rolling_window),
    show_progress = show_progress,
    compute_variance = compute_variance
  )

  # Extract individual components from returned dictionary
  propensity_scores <- result$propensity_scores
  total_loss <- result$total_loss
  lsd_max <- result$max_lsd
  lsd_mean <- result$mean_lsd

  # IPW weights (frequency weight ω*(p): 1 for ATE/Y, p for ATT)
  N <- length(propensity_scores)
  w_star <- if (ate_flag == 1L) rep(1, N) else propensity_scores
  ipw <- w_star / (Tr * propensity_scores + (1 - Tr) * (1 - propensity_scores))

  out <- list(
    fitted.values = propensity_scores,
    weights = ipw,
    loss = total_loss,
    lsd_train = list(
      lsd_max = lsd_max,
      lsd_mean = lsd_mean
    ),
    parameters = list(
      hidden_dim = hidden_dim,
      num_hidden_layers = num_hidden_layers,
      vae_lr = vae_lr,
      lr = lr,
      weight_decay = weight_decay,
      balance_lambda = balance_lambda,
      epsilon = epsilon
    ),
    stopping_criteria = list(
      lsd_threshold = lsd_threshold,
      rolling_window = rolling_window,
      max_epochs = max_epochs
    ),
    estimand   = estimand,
    ate_flag   = ate_flag,
    seed = seed,
    call = match.call(),
    formula = formula,
    Z = Z,
    Tr = Tr,
    ck = ck,
    h = h,
    rho = rho,
    kernel = kernel,
    K = K,
    ps_logistic = ps_log
  )
  
  # If Y was provided, store outcome + effect + variance
  if (!is.null(Y)) {
    out$Y <- Y
    out$effect <- result$effect
    out$se <- result$se
    out$ci <- c(lower = result$ci_lower, upper = result$ci_upper)
  }

  class(out) <- "lbc_net"  # Assign class to make it compatible with S3 methods
  return(out)

}
