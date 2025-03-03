#' Compute Adaptive Bandwidth for Kernel Smoothing
#'
#' @description This function calculates adaptive bandwidth values (`h`) for kernel smoothing
#' based on a given span (`rho`) and estimated propensity scores (`p`).
#' It is used in \code{\link{lbc_net}} but can also be applied independently
#' for bandwidth selection in other kernel-based methods.
#'
#' @param rho Numeric. The span (proportion of data points) used to determine the adaptive bandwidth.
#'   Ensures a sufficient local sample size for accurate balance estimation.
#' @param ck Numeric vector. Pre-specified kernel center points, strictly between 0 and 1.
#' @param p Numeric vector. Estimated propensity scores, with values strictly between 0 and 1.
#'   Typically generated from logistic regression but can be user-supplied.
#'
#' @return A numeric vector of bandwidth values (`h`), corresponding to each `ck`.
#'
#' @details
#' The adaptive bandwidths (`h`) ensure that each local region contains approximately
#' \code{rho * N} observations, where \code{N} is the total number of observations.
#' This helps maintain stable kernel-based local balance estimation.
#'
#' @examples
#' # Simulated dataset
#' set.seed(123)
#' N <- 500
#' Z <- as.data.frame(matrix(rnorm(N * 5), ncol = 5))
#' colnames(Z) <- paste0("X", 1:5)
#' T <- rbinom(N, 1, 0.5)  # Binary treatment assignment
#'
#' # Logistic regression for propensity score estimation
#' log.fit <- glm(T ~ ., data = Z, family = "binomial")
#' p <- log.fit$fitted.values  # Extract estimated propensity scores
#'
#' # Compute bandwidths
#' ck <- seq(0.01, 0.99, 0.01)  # Kernel center points
#' rho <- 0.15  # Span
#' h <- span_bw(rho, ck, p)
#' print(h)
#'
#' @export
span_bw <- function(rho, ck, p) {
  # Input validation
  if (!is.numeric(rho) || rho <= 0 || rho > 1) {
    stop("Error: 'rho' must be a numeric value between 0 and 1.")
  }

  if (!is.numeric(ck) || any(ck <= 0) || any(ck >= 1)) {
    stop("Error: 'ck' must be a numeric vector with values in (0,1).")
  }

  if (!is.numeric(p) || any(p <= 0) || any(p >= 1)) {
    stop("Error: 'p' must be a numeric vector with values strictly between 0 and 1.")
  }

  # Number of propensity scores
  N <- length(p)

  # Initialize bandwidth vector
  h <- numeric(length(ck))

  # Compute bandwidth for each ck
  for (i in seq_along(ck)) {
    d <- abs(ck[i] - p)  # Compute distance between ck[i] and each propensity score
    d_sort <- sort(d)    # Sort distances
    h[i] <- d_sort[ceiling(N * rho)]  # Select the rho*N-th smallest distance
  }

  return(h)
}
