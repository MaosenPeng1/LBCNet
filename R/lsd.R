#' Local Standardized Mean Difference (LSD) Calculation
#'
#' @description Computes the Local Standardized Mean Difference (LSD) for assessing local balance in causal inference.
#' The LSD measures the standardized mean difference for covariates at pre-specified grid points `ck` using kernel-based local inverse probability weighting of propensity scores.
#'
#' @param object An optional object of class `lbc_net`. If provided, extracts `Z`, `Tr`, `ps`, `ck`, `h`, and `kernel`.
#' @param Z A matrix or data frame of covariates. Required if `object` is not provided.
#' @param Tr A binary vector indicating treatment assignment (1 for treatment, 0 for control). Required if `object` is not provided.
#' @param ps A vector of propensity scores. Required if `object` is not provided.
#' @param ck A numeric vector of pre-specified grid points for local balance assessment. If NULL, it is automatically calculated.
#' @param h A numeric vector of bandwidths. If NULL, it is automatically calculated.
#' @param K The number of grid points to use if `ck` is not provided. Default is 99.
#' @param rho A scaling parameter used in bandwidth calculation. Default is 0.15.
#' @param ate_flag An integer (0 or 1) specifying the target estimand. The default is 1, which estimates the
#'   Average Treatment Effect (ATE) by weighting all observations equally. Setting it to 0 estimates the
#'   Average Treatment Effect on the Treated (ATT), where only treated units are fully weighted while control
#'   units are downweighted based on their propensity scores. See Details in \code{\link{lbc_net}}.
#' @param kernel The kernel function used. Options are "gaussian", "uniform", or "epanechnikov". Default is "gaussian".
#' @param ... Additional arguments passed to the specific method.
#'
#' @details
#' See \code{\link{lbc_net}} for details for local kernel weights and arguments `ck`, `h`, `K`, `rho`, and `kernel`.
#'
#' The formula for LSD follows the same structure as the Global Standardized Mean Difference (GSD) (\code{\link{gsd}}),
#' but the weights `W_i` are replaced with `W'_i`, Local Inverse Probability Weight (LIPW). LSD is expressed as a percentage in absolute value.
#'
#' Like GSD, LSD can be used for assessing balance in covariates, but LSD is specific to propensity score-based methods.
#'
#' @return  An object `lsd` containing LSD values (\%) for each covariate, which includes:
#'   \itemize{
#'         \item `LSD`: A matrix of LSD values for each covariate at each `ck`.
#'         \item `LSD_mean`: The mean absolute LSD value across all covariates.
#'         \item `LSD_max`: The maximum absolute LSD value.
#'       }
#' Other model components (e.g., `Z`, `Tr`) are accessible via `$`
#' or the recommended \code{\link[=getLBC.lsd]{getLBC}} function. While direct access (e.g., `fit$fitted.values`)
#' is possible, using `getLBC(fit, "LSD")` is recommended for stability and future-proofing.
#'
#' @examples
#' # Example with manually provided inputs
#' set.seed(123)
#' Z <- matrix(rnorm(200), nrow = 100, ncol = 2)
#' Tr <- rbinom(100, 1, 0.5)
#' ps <- runif(100, 0.1, 0.9)  # Simulated propensity scores
#'
#' # Compute LSD using manually provided inputs
#' lsd_result <- lsd(Z = Z, Tr = Tr, ps = ps, K = 99)
#' print(lsd_result)
#' summary(lsd_result)
#'
#' # Compute and visualize LSD results
#' plot(lsd_result)
#'
#' \dontrun{
#' # Fit LBC-Net model
#' model <- lbc_net(data = data, formula = Tr ~ X1 + X2 + X3 + X4)
#'
#' # Compute LSD from the fitted model for ATE.
#' lsd_fit <- lsd(model)
#' print(lsd_fit)
#' summary(lsd_result)
#'
#' # Visualize LSD results
#' plot(lsd_fit)
#' }
#' @importFrom stats glm
#' @export
lsd <- function(object = NULL, Z = NULL, Tr = NULL, ps = NULL, ck = NULL, h = NULL, K = 99, rho = 0.15, kernel = "gaussian", ate_flag = 1, ...) {
  if (!is.null(object)) {
    if (!inherits(object, "lbc_net")) {
      stop("Error: `object` must be of class 'lbc_net'.")
    }
    Tr <- getLBC(object, "Tr")
    Z <- getLBC(object, "Z")
    ps <- getLBC(object, "fitted.values")
    ck <- getLBC(object, "ck")
    h <- getLBC(object, "h")
    kernel <- getLBC(object, "kernel")
    ate_flag <- getLBC(object, "ate_flag")
  } else {

    if (is.null(Z) || is.null(Tr) || is.null(ps)) {
      stop("Error: Must provide `Z`, `Tr`, and `ps`, or an `lbc_net` object.")
    }

    # Calculate ps_log for ck/h calculation
    message("Calculating propensity scores for ck/h calculation...")
    log.fit <- glm(Tr ~ ., data = as.data.frame(Z), family = "binomial")
    ps_log <- log.fit$fitted.values

    # Auto-calculate `ck` and `h` if they were not provided
    if (is.null(ck)) {
      ck <- seq(1 / (K + 1), K / (K + 1), length.out = K)
    }
    if (is.null(h)) {
      h <- span_bw(rho, ck, ps_log)
    }

    if (!is.numeric(ck) || any(ck <= 0 | ck >= 1)) {
      stop("`ck` must be a numeric vector with values strictly between 0 and 1.")
    }

    if (!is.numeric(h) || length(h) != length(ck)) {
      stop("`h` must be a numeric vector of the same length as `ck`.")
    }
  }

  kernel_function <- function(x, kernel) {
    if (kernel == "gaussian") {
      return(1 / sqrt(2 * pi) * exp(-x^2 / 2))
    } else if (kernel == "uniform") {
      return(ifelse(abs(x) <= 1, 0.5, 0))
    } else if (kernel == "epanechnikov") {
      return(ifelse(abs(x) <= 1, 0.75 * (1 - x^2), 0))
    } else {
      stop("Invalid kernel specified. Choose 'gaussian', 'uniform', or 'epanechnikov'.")
    }
  }

  compute_lsd <- function(X_col, ate_flag=1) {
    N <- length(ps)
    LSD <- numeric(length(ck))

    w_star <- if (ate_flag == 1) rep(1, N) else ps

    for (i in seq_along(ck)) {
      w <- 1 / h[i] * kernel_function((ck[i] - ps) / h[i], kernel)
      W <- (w * w_star) / (Tr * ps + (1 - Tr) * (1 - ps))

      mu1 <- sum(Tr * W * X_col) / sum(Tr * W)
      mu0 <- sum((1 - Tr) * W * X_col) / sum((1 - Tr) * W)
      v1 <- sum(Tr * W * (X_col - mu1)^2) / sum(Tr * W)
      v0 <- sum((1 - Tr) * W * (X_col - mu0)^2) / sum((1 - Tr) * W)

      v1 <- max(v1, 1e-8)  # Apply small lower bound
      v0 <- max(v0, 1e-8)  # Apply small lower bound

      ess1 <- (sum(Tr * W))^2 / sum(Tr * W^2)
      ess0 <- (sum((1 - Tr) * W))^2 / sum((1 - Tr) * W^2)

      LSD[i] <- 100 * (mu1 - mu0) / sqrt((ess1 * v1 + ess0 * v0) / (ess1 + ess0))
    }
    return(abs(LSD))
  }

  if (is.matrix(Z) || is.data.frame(Z)) {
    LSD_values <- apply(as.matrix(Z), 2, compute_lsd, ate_flag = ate_flag)
  } else {
    LSD_values <- compute_lsd(Z, ate_flag = ate_flag)
  }

  out <- list(LSD = LSD_values, LSD_mean = mean(abs(LSD_values)), LSD_max = max(abs(LSD_values)), ck = ck, h = h, Z = Z, Tr = Tr, K = K, rho = rho, kernel = kernel, ate_flag = ate_flag)
  class(out) <- "lsd"
  return(out)

}


