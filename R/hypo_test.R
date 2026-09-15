#' Hypothesis Tests for ATE Marginal Treatment Means
#'
#' @description Performs global, pairwise, or custom linear hypothesis tests
#'   using the estimated marginal treatment means and their full stored joint
#'   influence-function covariance matrix.
#' @md
#' @param object An ATE inference object returned by \code{\link{lbc_net}} or
#'   \code{\link{m_lbcnet}} with an outcome supplied and variance estimation
#'   enabled. The object must contain `means` and `covariance`; a propensity
#'   score fit alone or a numeric result from \code{\link{est_effect}} is
#'   insufficient. Older binary fits without these fields must be refitted.
#' @param type One of `"all"` (default), `"global"`, `"pairwise"`, or `"custom"`.
#' @param C Numeric, finite, full-row-rank contrast matrix for `type = "custom"`.
#'   There must be one column per treatment. Unnamed columns follow the row
#'   order of `object$means`. If column names are supplied, they must identify
#'   all treatment labels and are aligned to that order.
#' @param rhs Numeric, finite right-hand side of the custom null hypothesis,
#'   with length `nrow(C)`. Defaults to a zero vector. `C` and `rhs` are only
#'   accepted for `type = "custom"`.
#' @param alpha Significance level used for pairwise confidence intervals,
#'   strictly between zero and one. Default is `0.05` (95% intervals).
#' @param p_adjust Pairwise p-value adjustment method accepted by
#'   [stats::p.adjust()], including `"none"` (default) and `"holm"`.
#'   Adjustment applies to the family of all pairwise comparisons and does
#'   not change the global/custom test or the confidence intervals.
#'
#' @details
#' This implementation applies to the current ATE inference framework.
#' Let \eqn{\hat\mu} contain the marginal treatment mean estimates and let
#' \eqn{V_\mu} be their stored joint covariance matrix. No covariance is
#' reconstructed from marginal standard errors.
#'
#' The global null is \eqn{H_0: \mu_1 = \cdots = \mu_L}. It uses
#' `cbind(diag(L - 1), -rep(1, L - 1))`, comparing each of the first
#' \eqn{L-1} means with the last. The alternative is that at least one mean
#' differs. The Wald statistic is asymptotically chi-square with \eqn{L-1}
#' degrees of freedom. For two groups it equals the squared pairwise z
#' statistic, up to numerical precision.
#'
#' Custom tests evaluate \eqn{H_0: C\mu = \mathrm{rhs}} using
#' \deqn{W = (C\hat\mu - \mathrm{rhs})^T
#'   (C V_\mu C^T)^{-1}(C\hat\mu - \mathrm{rhs}).}
#' The reference distribution is chi-square with `rank(C)` degrees of freedom.
#' Covariances are symmetrized locally to remove floating point asymmetry;
#' the stored estimator is not changed. A singular or numerically singular
#' contrast covariance raises an error; no generalized inverse is used.
#'
#' Pairwise tests evaluate \eqn{H_0: \mu_t - \mu_s = 0} for each pair in
#' the stored treatment order, retaining the original labels. Their standard
#' errors use the full covariance, and their two-sided p-values use the
#' asymptotic standard normal distribution. Binary fits store means in order
#' `1, 0`, so the contrast remains treatment minus control, consistent with
#' the existing binary effect. M-LBCNet retains its fitted treatment order
#' and the direction of `pairwise_ate`.
#'
#' Confidence intervals use `qnorm(1 - alpha / 2)`. Existing fit intervals
#' use the rounded multiplier `1.96`; those stored intervals are preserved,
#' so their endpoints can differ slightly from these intervals at `alpha = 0.05`.
#'
#' Tests are attached automatically as `object$hypothesis_test` when an ATE
#' fit computes outcome inference. For M-LBCNet, tests are not calculated
#' when `compute_variance = FALSE`. If the contrast covariance is singular,
#' automatic testing warns and stores `hypothesis_test = NULL`, preserving
#' the completed fit and its estimates. An explicit `hypo_test()` call still
#' raises an error in that case. Calling this function again allows a
#' different test, confidence level, or multiplicity adjustment.
#'
#' @return For `type = "all"`, a list with `global` and `pairwise` data frames.
#'   For `"global"`, a one-row data frame containing `hypothesis`, `statistic`
#'   (Wald chi-square), `df`, and `p_value`. For `"pairwise"`, a data frame
#'   containing `treatment_1`, `treatment_2`, `estimate`, `se`, `statistic`
#'   (signed z), `p_value`, `ci_lower`, `ci_upper`, and `p_adjusted`.
#'   For `"custom"`, a list containing `statistic` (Wald chi-square), `df`,
#'   `p_value`, `contrast_estimate = as.vector(C %*% mu_hat - rhs)`, and
#'   `contrast_covariance = C %*% V_mu %*% t(C)`.
#' @examples
#' \dontrun{
#' binary_ate_result <- lbc_net(Z = Z, Tr = Tr, Y = Y, estimand = "ATE")
#' h <- hypo_test(binary_ate_result)
#' h$global
#' h$pairwise
#'
#' multi_ate_result <- m_lbcnet(Z = Z, Tr = treatment, Y = Y)
#' h <- hypo_test(multi_ate_result)
#' h$global
#' h$pairwise
#' multi_ate_result$hypothesis_test
#'
#' # One three-treatment mean versus the average of the other two
#' C <- matrix(c(1, -0.5, -0.5), nrow = 1)
#' hypo_test(multi_ate_result, type = "custom", C = C)
#' hypo_test(multi_ate_result, type = "custom", C = C, rhs = 1)
#' hypo_test(multi_ate_result, type = "pairwise", p_adjust = "holm")
#' }
#' @export
hypo_test <- function(object, type = c("all", "global", "pairwise", "custom"),
                      C = NULL, rhs = NULL, alpha = 0.05, p_adjust = "none") {
  type <- match.arg(type)
  if (!is.character(p_adjust) || length(p_adjust) != 1L || is.na(p_adjust) ||
      !p_adjust %in% stats::p.adjust.methods) {
    stop("`p_adjust` must be one of: ",
         paste(stats::p.adjust.methods, collapse = ", "), ".", call. = FALSE)
  }
  if (!is.numeric(alpha) || is.complex(alpha) || length(alpha) != 1L ||
      !is.finite(alpha) || alpha <= 0 || alpha >= 1) {
    stop("`alpha` must be a finite number strictly between 0 and 1.",
         call. = FALSE)
  }
  if (!inherits(object, c("lbc_net", "m_lbcnet"))) {
    stop("`object` must be an ATE inference object from lbc_net() or m_lbcnet().",
         call. = FALSE)
  }
  if (!identical(object$estimand, "ATE")) {
    stop("hypo_test() supports ATE inference only; fit with estimand = 'ATE'.",
         call. = FALSE)
  }
  if (is.null(object$means) || is.null(object$covariance)) {
    stop("ATE marginal means and their full joint covariance are required. ",
         "Refit with an outcome Y and variance estimation enabled.",
         call. = FALSE)
  }
  means <- object$means
  if (!is.data.frame(means) ||
      !all(c("treatment", "estimate") %in% names(means))) {
    stop("`object$means` must contain treatment and estimate columns.",
         call. = FALSE)
  }
  mu_hat <- means$estimate
  treatment_levels <- means$treatment
  labels <- as.character(treatment_levels)
  n_treatments <- nrow(means)
  if (n_treatments < 2L || !is.numeric(mu_hat) || is.complex(mu_hat) ||
      any(!is.finite(mu_hat))) {
    stop("`object$means` must contain at least two finite numeric estimates.",
         call. = FALSE)
  }
  if (anyNA(labels) || any(!nzchar(labels)) || anyDuplicated(labels)) {
    stop("Treatment labels in `object$means` must be nonmissing and unique.",
         call. = FALSE)
  }
  V_mu <- object$covariance
  if (!is.matrix(V_mu) || !is.numeric(V_mu) || is.complex(V_mu) ||
      !identical(dim(V_mu), c(n_treatments, n_treatments))) {
    stop("`object$covariance` must be a numeric square matrix with one row ",
         "and column per treatment mean.", call. = FALSE)
  }
  if (any(!is.finite(V_mu))) {
    stop("The joint covariance must be finite. Refit with variance ",
         "estimation enabled (compute_variance = TRUE for m_lbcnet).",
         call. = FALSE)
  }
  if (!is.null(rownames(V_mu)) || !is.null(colnames(V_mu))) {
    if (anyDuplicated(rownames(V_mu)) || anyDuplicated(colnames(V_mu)) ||
        !setequal(rownames(V_mu), labels) ||
        !setequal(colnames(V_mu), labels)) {
      stop("Joint covariance row and column names must match the treatment labels.",
           call. = FALSE)
    }
    V_mu <- V_mu[labels, labels, drop = FALSE]
  }
  V_mu <- (V_mu + t(V_mu)) / 2

  if (type == "custom") {
    if (!is.null(colnames(C))) {
      if (anyDuplicated(colnames(C)) || !setequal(colnames(C), labels)) {
        stop("`C` column names must match the treatment labels.", call. = FALSE)
      }
      C <- C[, labels, drop = FALSE]
    }
    return(.wald_test_mu(mu_hat, V_mu, C, rhs))
  }
  if (!is.null(C) || !is.null(rhs)) {
    stop("`C` and `rhs` are only used with type = 'custom'.", call. = FALSE)
  }

  result <- list()
  if (type %in% c("all", "global")) {
    C_global <- cbind(diag(n_treatments - 1L), -rep(1, n_treatments - 1L))
    global <- .wald_test_mu(mu_hat, V_mu, C_global)
    result$global <- data.frame(
      hypothesis = "All treatment-specific means are equal",
      statistic = global$statistic, df = global$df, p_value = global$p_value,
      stringsAsFactors = FALSE
    )
  }
  if (type %in% c("all", "pairwise")) {
    pairs <- utils::combn(seq_len(n_treatments), 2L)
    critical <- stats::qnorm(alpha / 2, lower.tail = FALSE)
    rows <- lapply(seq_len(ncol(pairs)), function(index) {
      treatment_1 <- pairs[1L, index]
      treatment_2 <- pairs[2L, index]
      contrast <- matrix(0, nrow = 1L, ncol = n_treatments)
      contrast[1L, treatment_1] <- 1
      contrast[1L, treatment_2] <- -1
      tested <- .wald_test_mu(mu_hat, V_mu, contrast)
      estimate <- tested$contrast_estimate[1L]
      se <- sqrt(tested$contrast_covariance[1L, 1L])
      z <- estimate / se
      data.frame(
        treatment_1 = treatment_levels[treatment_1],
        treatment_2 = treatment_levels[treatment_2],
        estimate = estimate, se = se, statistic = z,
        p_value = 2 * stats::pnorm(-abs(z)),
        ci_lower = estimate - critical * se,
        ci_upper = estimate + critical * se,
        check.names = FALSE, stringsAsFactors = FALSE
      )
    })
    result$pairwise <- do.call(rbind, rows)
    rownames(result$pairwise) <- NULL
    result$pairwise$p_adjusted <- stats::p.adjust(
      result$pairwise$p_value, method = p_adjust
    )
  }
  if (type == "all") result else result[[type]]
}


.wald_test_mu <- function(mu_hat, V_mu, C, rhs = NULL) {
  if (!is.matrix(C) || !is.numeric(C) || is.complex(C) ||
      nrow(C) < 1L || any(!is.finite(C))) {
    stop("`C` must be a finite numeric matrix with at least one row.",
         call. = FALSE)
  }
  if (ncol(C) != length(mu_hat)) {
    stop("`C` must have one column per treatment mean (", length(mu_hat), ").",
         call. = FALSE)
  }
  tolerance <- sqrt(.Machine$double.eps)
  rank <- qr(C, tol = tolerance)$rank
  if (rank != nrow(C)) {
    stop("`C` must have full row rank; the supplied contrasts are rank deficient.",
         call. = FALSE)
  }
  if (is.null(rhs)) rhs <- numeric(nrow(C))
  if (!is.numeric(rhs) || is.complex(rhs) || !is.null(dim(rhs)) ||
      length(rhs) != nrow(C) || any(!is.finite(rhs))) {
    stop("`rhs` must be a finite numeric vector with length nrow(C).",
         call. = FALSE)
  }
  difference <- as.vector(C %*% mu_hat - rhs)
  covariance <- C %*% V_mu %*% t(C)
  covariance <- (covariance + t(covariance)) / 2
  if (any(!is.finite(difference)) || any(!is.finite(covariance))) {
    stop("The contrast estimates and contrast covariance matrix must be finite.",
         call. = FALSE)
  }
  eigenvalues <- eigen(covariance, symmetric = TRUE, only.values = TRUE)$values
  if (min(eigenvalues) <= 0 ||
      min(eigenvalues) <= max(eigenvalues) * tolerance) {
    .wald_singular_error()
  }
  solved <- tryCatch(
    solve(covariance, difference, tol = tolerance),
    error = function(error) .wald_singular_error()
  )
  statistic <- as.numeric(crossprod(difference, solved))
  list(
    statistic = statistic, df = rank,
    p_value = stats::pchisq(statistic, df = rank, lower.tail = FALSE),
    contrast_estimate = difference, contrast_covariance = covariance
  )
}


.wald_singular_error <- function() {
  stop(errorCondition(
    "The contrast covariance matrix is singular or numerically singular (not positive definite); the Wald test cannot be computed.",
    class = "lbcnet_singular_contrast"
  ))
}


.attach_ate_hypothesis_test <- function(object) {
  result <- tryCatch(
    hypo_test(object),
    lbcnet_singular_contrast = function(error) {
      warning("ATE hypothesis tests are unavailable: ", conditionMessage(error),
              call. = FALSE)
      NULL
    }
  )
  object["hypothesis_test"] <- list(result)
  object
}
