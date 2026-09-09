#' Compute Global Standardized Mean Difference (GSD)
#'
#' @description Estimates the Global Standardized Mean Difference (GSD), a standardized mean difference
#' used for assessing balance in covariate distributions between treatment groups.
#' The GSD is reported as a percentage and is widely used in propensity score weighting methods.
#'
#' @param object An optional object of class `"lbc_net"` or `"m_lbcnet"`.
#'   For a binary fit, `Z`, `Tr`, and `weights` are extracted. For an
#'   M-LBCNet fit, treatment-versus-population and pairwise diagnostics are
#'   computed from the joint generalized propensity scores.
#' @param Z A numeric matrix, data frame, or vector of covariates. Required if `object` is not provided.
#' @param Tr A numeric vector (0/1) indicating treatment assignment. Required if `object` is not provided.
#' @param ps A numeric vector of propensity scores (\eqn{0 < ps < 1}). 
#'   Used to compute weights as:
#'   \deqn{\frac{w^*(ps)}{Tr \cdot ps + (1 - Tr) \cdot (1 - ps)}.}
#'   The argument \code{ATE} must be specified: if \code{ATE = 1}, then \eqn{w^*(ps) = 1}; 
#'   if \code{ATE = 0}, then \eqn{w^*(ps) = ps}. Ignored if \code{wt} is provided.
#' @param wt A numeric vector of inverse probability weights (IPW) or other balancing weights. If provided, `ps` is ignored.
#' @param ate_flag An integer (0 or 1) specifying the target estimand. The default is 1, which estimates the
#'   Average Treatment Effect (ATE) by weighting all observations equally. Setting it to 0 estimates the
#'   Average Treatment Effect on the Treated (ATT), where only treated units are fully weighted while control
#'   units are downweighted based on their propensity scores. Ignored if \code{wt} is provided.
#'   See \code{\link{lbc_net}} for more information on ATT, ATE, and their corresponding weighting schemes.
#' @param ... Additional arguments passed to the specific method.
#'
#' @return For a binary treatment, a numeric vector containing GSD values for
#'   each covariate. For an `"m_lbcnet"` object, a list with two data frames:
#'   `versus_population`, the primary diagnostic, and `pairwise`, the secondary
#'   diagnostic. Original treatment labels are retained.
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
#' For M-LBCNet, arm \eqn{t} uses weights
#' \eqn{a_{it}=R_{it}/\pi_{it}} and the common ATE population uses
#' \eqn{b_i=1}. Pairwise comparisons replace \eqn{b_i} with
#' \eqn{a_{is}=R_{is}/\pi_{is}}. Both use the implemented binary
#' weighted-variance convention (division by total weight), effective sample
#' sizes \eqn{m=(\sum_i w_i)^2/\sum_i w_i^2}, and the pooled standardizer
#' shown above. M-LBCNet GSD values are reported in absolute value.
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
gsd <- function(object = NULL, Z = NULL, Tr = NULL, ps = NULL, wt = NULL, ate_flag = 1, ...) {
  if (!is.null(object) && inherits(object, "m_lbcnet")) {
    return(.m_lbcnet_gsd(object))
  }

  # Extract from `lbc_net` object if provided
  if (!is.null(object)) {
    if (!inherits(object, "lbc_net")) {
      stop("Error: `object` must be of class 'lbc_net'.")
    }
    Z <- getLBC(object, "Z")  # Extract covariates
    Tr <- getLBC(object, "Tr")  # Extract treatment
    wt <- getLBC(object, "weights")  # Extract weights
    ate_flag <- getLBC(object, "ate_flag")
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
    w_star <- if (ate_flag == 1) rep(1, N) else ps
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


.m_lbcnet_standardized_difference <- function(
    Z, weights_1, weights_2, mass_floor = NULL, variance_floor = NULL,
    invalid_value = NA_real_) {
  Z <- as.matrix(Z)
  mass_1 <- sum(weights_1)
  mass_2 <- sum(weights_2)
  squared_mass_1 <- sum(weights_1^2)
  squared_mass_2 <- sum(weights_2^2)

  if (!is.null(mass_floor)) {
    valid <- all(is.finite(c(
      mass_1, mass_2, squared_mass_1, squared_mass_2
    ))) && mass_1 > mass_floor && mass_2 > mass_floor &&
      squared_mass_1 > 0 && squared_mass_2 > 0
    if (!valid) {
      return(stats::setNames(rep(invalid_value, ncol(Z)), colnames(Z)))
    }
  }

  mean_1 <- colSums(sweep(Z, 1L, weights_1, `*`)) / mass_1
  mean_2 <- colSums(sweep(Z, 1L, weights_2, `*`)) / mass_2
  centered_1 <- sweep(Z, 2L, mean_1, `-`)
  centered_2 <- sweep(Z, 2L, mean_2, `-`)
  variance_1 <- colSums(sweep(centered_1^2, 1L, weights_1, `*`)) / mass_1
  variance_2 <- colSums(sweep(centered_2^2, 1L, weights_2, `*`)) / mass_2
  if (!is.null(variance_floor)) {
    variance_1 <- pmax(variance_1, variance_floor)
    variance_2 <- pmax(variance_2, variance_floor)
  }

  ess_1 <- mass_1^2 / squared_mass_1
  ess_2 <- mass_2^2 / squared_mass_2
  if (!is.null(mass_floor) && !all(is.finite(c(ess_1, ess_2)))) {
    return(stats::setNames(rep(invalid_value, ncol(Z)), colnames(Z)))
  }
  pooled_variance <- (
    ess_1 * variance_1 + ess_2 * variance_2
  ) / (ess_1 + ess_2)
  values <- 100 * abs(mean_1 - mean_2) / sqrt(pooled_variance)
  if (!is.null(mass_floor)) {
    values[!is.finite(values)] <- invalid_value
  }
  stats::setNames(as.numeric(values), colnames(Z))
}


.m_lbcnet_gsd <- function(object) {
  Z <- as.matrix(object$Z)
  gps <- as.matrix(object$fitted.values)
  treatment_code <- object$Tr_code
  treatment_levels <- object$treatment_levels
  n_treatments <- object$n_treatments
  covariates <- colnames(Z)
  population_weights <- rep(1, nrow(Z))

  versus_population <- do.call(rbind, lapply(
    seq_len(n_treatments),
    function(treatment_index) {
      treatment_weights <-
        (treatment_code == treatment_index - 1L) /
        gps[, treatment_index]
      data.frame(
        treatment = rep(treatment_levels[treatment_index], ncol(Z)),
        covariate = covariates,
        gsd = unname(.m_lbcnet_standardized_difference(
          Z, treatment_weights, population_weights
        )),
        check.names = FALSE,
        stringsAsFactors = FALSE
      )
    }
  ))
  rownames(versus_population) <- NULL

  pairwise_rows <- list()
  row_index <- 1L
  for (treatment_1 in seq_len(n_treatments - 1L)) {
    weights_1 <-
      (treatment_code == treatment_1 - 1L) / gps[, treatment_1]
    for (treatment_2 in seq.int(treatment_1 + 1L, n_treatments)) {
      weights_2 <-
        (treatment_code == treatment_2 - 1L) / gps[, treatment_2]
      pairwise_rows[[row_index]] <- data.frame(
        treatment_1 = rep(treatment_levels[treatment_1], ncol(Z)),
        treatment_2 = rep(treatment_levels[treatment_2], ncol(Z)),
        covariate = covariates,
        gsd = unname(.m_lbcnet_standardized_difference(
          Z, weights_1, weights_2
        )),
        check.names = FALSE,
        stringsAsFactors = FALSE
      )
      row_index <- row_index + 1L
    }
  }
  pairwise <- do.call(rbind, pairwise_rows)
  rownames(pairwise) <- NULL

  list(
    versus_population = versus_population,
    pairwise = pairwise
  )
}
