#' Compute Global Standardized Mean Difference (GSD)
#'
#' @description Estimates the Global Standardized Mean Difference (GSD), a standardized mean difference
#' used for assessing balance in covariate distributions between treatment groups.
#' The GSD is reported as a percentage and is widely used in propensity score weighting methods.
#'
#' @param object An optional object of class `"lbc_net"`, from which `Z`, `Tr`, and `weights` are extracted.
#' @param Z A numeric matrix, data frame, or vector of covariates. Required if `object` is not provided.
#' @param Tr A numeric vector (0/1) indicating treatment assignment. Required if `object` is not provided.
#' @param ps A numeric vector of propensity scores (\eqn{0 < ps < 1}). 
#'   Used to compute weights as:
#'   \deqn{\frac{w^*(ps)}{Tr \cdot ps + (1 - Tr) \cdot (1 - ps)}.}
#'   The argument \code{ATE} must be specified: if \code{ATE = 1}, then \eqn{w^*(ps) = 1}; 
#'   if \code{ATE = 0}, then \eqn{w^*(ps) = ps}. Ignored if \code{wt} is provided.
#' @param wt A numeric vector of inverse probability weights (IPW) or other balancing weights. If provided, `ps` is ignored.
#' @param ATE An integer (0 or 1) specifying the target estimand. The default is 1, which estimates the
#'   Average Treatment Effect (ATE) by weighting all observations equally. Setting it to 0 estimates the
#'   Average Treatment Effect on the Treated (ATT), where only treated units are fully weighted while control
#'   units are downweighted based on their propensity scores. Ignored if \code{wt} is provided.
#'   See \code{\link{lbc_net}} for more information on ATT, ATE, and their corresponding weighting schemes.
#' @param ... Additional arguments passed to the specific method.
#'
#' @return A numeric vector containing GSD values for each covariate.
#'
#' @details
#' \strong{Definition of GSD}:
#'
#' The GSD measures covariate balance across treatment groups:
#' \deqn{
#' GSD = \frac{| \mu_1 - \mu_0 | }{ \sqrt{ ( m_1 v_1 + m_0 v_0 )/(m_1 + m_0) } } \times 100\%
#' }
#' where:
#' 
#' - \eqn{\mu_1} and \eqn{\mu_0} are the IPTW-weighted means for the treated and control groups:
#'   \deqn{
#'   \mu_1 = \frac{\sum_{i=1}^{N} T_i W_i X_i }{ \sum_{i=1}^{N} T_i W_i }, \quad
#'   \mu_0 = \frac{\sum_{i=1}^{N} (1-T_i) W_i X_i }{ \sum_{i=1}^{N} (1-T_i) W_i }.
#'   }
#'   
#' - \eqn{v_1} and \eqn{v_0} are the corresponding weighted variances:
#'   \deqn{
#'   v_1 = \frac{\sum_{i=1}^{N} T_i W_i (X_i - \mu_1)^2 }{ \sum_{i=1}^{N} T_i W_i - 1 }, \quad
#'   v_0 = \frac{\sum_{i=1}^{N} (1-T_i) W_i (X_i - \mu_0)^2 }{ \sum_{i=1}^{N} (1-T_i) W_i - 1 }.
#'   }
#'   
#' - \eqn{m_1} and \eqn{m_0} are the effective sample sizes (ESS) of the treated and control groups:
#'   \deqn{
#'   m_1 = \frac{ (\sum_{i=1}^{N} T_i W_i)^2 }{ \sum_{i=1}^{N} T_i W_i^2 }, \quad
#'   m_0 = \frac{ (\sum_{i=1}^{N} (1-T_i) W_i)^2 }{ \sum_{i=1}^{N} (1-T_i) W_i^2 }.
#'   }
#'
#' Automatic Extraction from `lbc_net` Object if an `lbc_net` object is provided.
#'
#' @examples
#'
#' # Example with manually provided inputs
#' set.seed(123)
#' Z <- matrix(rnorm(200), nrow = 100, ncol = 2)
#' Tr <- rbinom(100, 1, 0.5)
#' ps <- runif(100, 0.1, 0.9)  # Simulated propensity scores
#'
#' # Compute GSD using propensity scores
#' gsd(Z = Z, Tr = Tr, ps = ps)
#'
#' # Compute GSD using weights
#' wt <- 1 / (Tr * ps + (1 - Tr) * (1 - ps))  # Convert ps to weights
#' gsd(Z = Z, Tr = Tr, wt = wt)
#'
#' \dontrun{
#' # Example with an lbc_net object
#' model <- lbc_net(data = data, formula = Tr ~ X1 + X2 + X3 + X4)
#' gsd(model)
#' }
#' @export
gsd <- function(object = NULL, Z = NULL, Tr = NULL, ps = NULL, wt = NULL, ATE = 1, ...) {
  # Extract from `lbc_net` object if provided
  if (!is.null(object)) {
    if (!inherits(object, "lbc_net")) {
      stop("Error: `object` must be of class 'lbc_net'.")
    }
    Z <- getLBC(object, "Z")  # Extract covariates
    Tr <- getLBC(object, "Tr")  # Extract treatment
    wt <- getLBC(object, "weights")  # Extract weights
    ATE <- getLBC(object, "ATE")
  }

  # Ensure required inputs are provided
  if (is.null(Z) || is.null(Tr)) {
    stop("Error: Must provide `Z` (covariates) and `Tr` (treatment assignment), or an `lbc_net` object.")
  }
  
  # Coerce Z to a matrix and handle vectors
  if (is.vector(Z)) {
    Z <- matrix(Z, ncol = 1)
    colnames(Z) <- "V1"
  }
  
  if (is.data.frame(Z)) {
    Z <- as.matrix(Z)
  }
  
  if (!is.matrix(Z)) {
    stop("Error: `Z` must be a numeric vector, matrix, or data frame.")
  }
  
  # Add column names if missing
  if (is.null(colnames(Z))) {
    colnames(Z) <- paste0("V", seq_len(ncol(Z)))
  }

  # Convert propensity scores to weights if necessary
  if (!is.null(ps) && is.null(wt)) {
    if (any(ps <= 0 | ps >= 1)) {
      stop("Error: `ps` (propensity scores) must be strictly between 0 and 1.")
    }
    N <- length(ps)
    w_star <- if (ATE == 1) rep(1, N) else ps
    wt <- w_star / (Tr * ps + (1 - Tr) * (1 - ps))
  }

  # Ensure weights are available
  if (is.null(wt)) {
    stop("Error: Must provide either `wt` (weights) or `ps` (propensity scores).")
  }

  # Compute GSD for each covariate
  compute_gsd <- function(Z_col) {
    mu1 <- sum(Tr * wt * Z_col) / sum(Tr * wt)
    mu0 <- sum((1 - Tr) * wt * Z_col) / sum((1 - Tr) * wt)
    v1 <- sum(Tr * wt * (Z_col - mu1)^2) / sum(Tr * wt)
    v0 <- sum((1 - Tr) * wt * (Z_col - mu0)^2) / sum((1 - Tr) * wt)
    ess1 <- (sum(Tr * wt))^2 / sum(Tr * wt^2)
    ess0 <- (sum((1 - Tr) * wt))^2 / sum((1 - Tr) * wt^2)

    100 * (mu1 - mu0) / sqrt((ess1 * v1 + ess0 * v0) / (ess1 + ess0))
  }

  # Apply over columns if `Z` is a matrix or data frame
  if (is.matrix(Z) || is.data.frame(Z)) {
    return(apply(as.matrix(Z), 2, compute_gsd))
  } else {
    return(compute_gsd(Z))  # Single covariate case (vector)
  }
  
}
