#' Fit M-LBCNet for a Multi-Valued Treatment
#'
#' @description
#' Fits LBCNet for a categorical treatment with three or more
#' observed levels. The estimator uses one shared neural network and a
#' simplex-preserving softmax generalized propensity score (GPS).
#'
#' @details
#' For arm \eqn{t}, LBCNet balances that treatment against the population. 
#' The local balance contribution is
#' \deqn{\omega(c_k,\pi_i^{(t)})
#' \{R_i^{(t)}/\pi_i^{(t)}-1\}Z_i,}
#' and componentwise calibration uses
#' \deqn{\omega(c_k,\pi_i^{(t)})
#' \{R_i^{(t)}-\pi_i^{(t)}\}/\{c_k(1-c_k)\}.}
#' The calibration moment is multiplied by the square root of
#' \code{balance_lambda}, so its squared contribution has weight
#' \code{balance_lambda}. The stored training loss applies the common positive
#' optimizer scaling \eqn{N^2 Q^*}.
#'
#' Probabilities are
#' \deqn{\pi_i^{(t)}=\epsilon+(1-L\epsilon)
#' \operatorname{softmax}_t\{\eta(Z_i)\},}
#' preserving both the lower bound and row sums of one. The default grid has
#' \code{K = 19} centers from 0.05 through 0.95. Bandwidths are
#' treatment-specific. When bandwidths are selected adaptively, one preliminary
#' joint multinomial logistic GPS is fitted and used only with
#' \code{\link{span_bw}}. If that pilot has a numerical or convergence failure,
#' separate treatment-versus-rest binary logistic regressions are used instead
#' and a warning is issued. Neither pilot is used as the final M-LBCNet GPS.
#' Specifically, each row is selected as
#' \code{h[t, ] = span_bw(rho, ck, ps_preliminary[, t])}.
#'
#' The returned \code{fitted.values} is an \eqn{N} by \eqn{L} matrix and each
#' observed ATE weight is \eqn{1/\pi_i^{(T_i)}}. With an outcome, the function
#' estimates every marginal potential-outcome mean and all pairwise ATEs. Joint
#' influence-function inference uses one shared nuisance-parameter influence
#' function, a different sensitivity row for each mean, and retains the full
#' covariance matrix.
#'
#' Training diagnostics and the rolling early-stopping rule use only
#' treatment-versus-population LSD. This is the same quantity reported
#' publicly by \code{lsd(fit)$versus_population}. The secondary pairwise LSD
#' is an on-demand public diagnostic and is not used for optimization or
#' stopping.
#'
#' @param data Optional data frame containing variables in \code{formula}.
#' @param formula Optional treatment-on-covariates formula. When supplied,
#'   \code{data} must also be supplied.
#' @param Z Numeric covariate vector, matrix, or data frame used for direct input.
#' @param Tr Factor, character, integer, or numeric categorical treatment.
#'   Observed factor levels retain factor-level order; other labels use sorted
#'   deterministic order.
#' @param Y Optional finite numeric outcome. If \code{NULL}, only the GPS and
#'   training/balance diagnostics are returned.
#' @param K Positive number of local centers; default 19.
#' @param rho Span in \eqn{(0,1]} used by \code{\link{span_bw}}.
#' @param na.action Missing-data action; default \code{\link[stats]{na.fail}}.
#' @param gpu Nonnegative CUDA device index. CPU is used if CUDA is unavailable.
#' @param show_progress Logical; display Python training progress.
#' @param ... Tuning arguments \code{ck}, \code{h}, \code{kernel}, \code{seed},
#'   \code{hidden_dim}, \code{num_hidden_layers}, \code{vae_epochs},
#'   \code{vae_lr}, \code{max_epochs}, \code{lr}, \code{weight_decay},
#'   \code{balance_lambda}, \code{alpha}, \code{epsilon},
#'   \code{lsd_threshold}, \code{rolling_window}, and
#'   \code{compute_variance}. A supplied \code{h} may be a length-\code{K}
#'   vector or an \eqn{L} by \eqn{K} matrix and skips pilot estimation.
#' @param setup_lbcnet_args List passed to \code{\link{setup_lbcnet}} when
#'   Python is not already set up.
#'
#' @return An S3 object of class \code{"m_lbcnet"}. The
#'   \code{bandwidth_pilot_method} component records \code{"multinomial"},
#'   \code{"treatment_vs_rest"}, or \code{"not_used"} when \code{h} was
#'   supplied. The \code{ps_preliminary} component contains the pilot GPS used
#'   for adaptive selection and is \code{NULL} when \code{h} was supplied. With
#'   an outcome the object also includes \code{means}, \code{covariance},
#'   \code{pairwise_ate}, and, when variance is enabled,
#'   \code{influence_functions} and \code{hypothesis_test}. The latter is
#'   produced by \code{\link{hypo_test}} using the full joint covariance and
#'   contains global equality and pairwise ATE tests. It is not computed when
#'   \code{compute_variance = FALSE}.
#'
#' @examples
#' \dontrun{
#' set.seed(2026)
#' n <- 90
#' Z <- matrix(rnorm(n * 2), ncol = 2,
#'             dimnames = list(NULL, c("x1", "x2")))
#' Tr <- factor(rep(c("A", "B", "C"), each = n / 3),
#'              levels = c("A", "B", "C"))
#' Y <- 1 + as.numeric(Tr) + Z[, 1] + rnorm(n)
#' fit <- m_lbcnet(
#'   Z = Z, Tr = Tr, Y = Y, hidden_dim = 8,
#'   vae_epochs = 10, max_epochs = 200, show_progress = FALSE
#' )
#' fit$fitted.values
#' fit$means
#' fit$pairwise_ate
#' }
#'
#' @importFrom nnet multinom
#' @importFrom stats glm model.frame model.matrix model.response na.fail
#' @export
m_lbcnet <- function(data = NULL, formula = NULL, Z = NULL, Tr = NULL,
                     Y = NULL, K = 19, rho = 0.15, na.action = na.fail,
                     gpu = 0, show_progress = TRUE, ...,
                     setup_lbcnet_args = list()) {
  matched_call <- match.call()
  args <- list(...)

  ck <- if (!is.null(args$ck)) args$ck else NULL
  h <- if ("h" %in% names(args)) args$h else NULL
  kernel <- if (is.null(args$kernel)) "gaussian" else args$kernel
  seed <- if (is.null(args$seed)) 100L else args$seed
  hidden_dim <- if (is.null(args$hidden_dim)) 100L else args$hidden_dim
  num_hidden_layers <- if (is.null(args$num_hidden_layers)) 1L else
    args$num_hidden_layers
  vae_epochs <- if (is.null(args$vae_epochs)) 250L else args$vae_epochs
  vae_lr <- if (is.null(args$vae_lr)) 0.01 else args$vae_lr
  max_epochs <- if (is.null(args$max_epochs)) 5000L else args$max_epochs
  lr <- if (is.null(args$lr)) 0.05 else args$lr
  weight_decay <- if (is.null(args$weight_decay)) 1e-5 else args$weight_decay
  balance_lambda <- if (is.null(args$balance_lambda)) 1 else args$balance_lambda
  alpha <- if (is.null(args$alpha)) 0.01 else args$alpha
  epsilon <- if (is.null(args$epsilon)) 0.001 else args$epsilon
  lsd_threshold <- if (is.null(args$lsd_threshold)) 2 else args$lsd_threshold
  rolling_window <- if (is.null(args$rolling_window)) 5L else args$rolling_window
  compute_variance <- if (is.null(args$compute_variance)) TRUE else
    args$compute_variance

  input <- .m_lbcnet_prepare_inputs(
    data, formula, Z, Tr, Y, na.action
  )
  Z <- input$Z
  Tr <- input$Tr
  Y <- input$Y

  coding <- .m_lbcnet_treatment_coding(Tr)
  treatment_levels <- coding$levels
  Tr_code <- coding$code
  n_treatments <- length(treatment_levels)
  treatment_names <- as.character(treatment_levels)
  if (n_treatments * epsilon >= 1) {
    stop("'n_treatments * epsilon' must be strictly less than 1.")
  }

  if (is.null(ck)) {
    ck <- seq_len(as.integer(K)) / (as.integer(K) + 1)
  } else {
    if (!is.numeric(ck) || any(ck <= 0 | ck >= 1)) {
      stop("'ck' must be a numeric vector strictly inside (0, 1).")
    }
    ck <- as.numeric(ck)
  }
  K <- as.integer(length(ck))

  ps_preliminary <- NULL
  bandwidth_pilot_method <- "not_used"
  if (is.null(h)) {
    pilot <- .m_lbcnet_bandwidth_pilot(
      Z, Tr_code, treatment_names
    )
    ps_preliminary <- pilot$probabilities
    bandwidth_pilot_method <- pilot$method
    h <- do.call(rbind, lapply(
      seq_len(n_treatments),
      function(index) span_bw(rho, ck, ps_preliminary[, index])
    ))
  } else if (is.numeric(h) && is.null(dim(h))) {
    if (length(h) != K) {
      stop("A vector 'h' must have length K (the length of 'ck').")
    }
    h <- matrix(rep(as.numeric(h), each = n_treatments),
                nrow = n_treatments)
  } else if (is.matrix(h) && is.numeric(h)) {
    if (!identical(dim(h), c(n_treatments, K))) {
      stop("A matrix 'h' must have dimensions n_treatments by K.")
    }
    h <- unname(h)
  } else {
    stop("'h' must be NULL, a length-K vector, or an L by K matrix.")
  }
  dimnames(h) <- list(treatment_names, paste0("c", seq_len(K)))

  # Ensure Python is properly configured before running setup
  if (!reticulate::py_available(initialize = FALSE)) {
    message("Python environment is not set up. Running `setup_lbcnet()`...")
    do.call(setup_lbcnet, setup_lbcnet_args)
  } else {
    message("Python is already set up. Skipping `setup_lbcnet()`.")
  }

  module <- reticulate::import_from_path(
    "m_lbcnet", path = system.file("python", package = "LBCNet")
  )

  data_df <- as.data.frame(Z, check.names = FALSE)
  treatment_column <- .m_lbcnet_unique_name(
    ".m_lbcnet_treatment", names(data_df)
  )
  data_df[[treatment_column]] <- Tr_code
  outcome_column <- NULL
  if (!is.null(Y)) {
    outcome_column <- .m_lbcnet_unique_name(
      ".m_lbcnet_outcome", names(data_df)
    )
    data_df[[outcome_column]] <- as.numeric(Y)
  }

  result <- module$run_m_lbcnet(
    data_df = data_df,
    Z_columns = colnames(Z),
    T_column = treatment_column,
    Y_column = outcome_column,
    n_treatments = as.integer(n_treatments),
    ck = as.numeric(ck),
    h = unname(h),
    kernel = kernel,
    gpu = as.integer(gpu),
    seed = as.integer(seed),
    hidden_dim = as.integer(hidden_dim),
    # Same depth convention as the current binary R wrapper.
    num_layers = as.integer(num_hidden_layers + 1L),
    vae_epochs = as.integer(vae_epochs),
    vae_lr = as.numeric(vae_lr),
    max_epochs = as.integer(max_epochs),
    lr = as.numeric(lr),
    weight_decay = as.numeric(weight_decay),
    balance_lambda = as.numeric(balance_lambda),
    epsilon = as.numeric(epsilon),
    lsd_threshold = as.numeric(lsd_threshold),
    alpha = as.numeric(alpha),
    rolling_window = as.integer(rolling_window),
    show_progress = show_progress,
    compute_variance = isTRUE(compute_variance) && !is.null(Y)
  )

  fitted_values <- as.matrix(result$propensity_scores)
  storage.mode(fitted_values) <- "double"
  if (!identical(dim(fitted_values), c(nrow(Z), n_treatments))) {
    stop("Python returned a GPS matrix with unexpected dimensions.")
  }
  colnames(fitted_values) <- treatment_names
  if (any(!is.finite(fitted_values)) ||
      any(fitted_values <= 0) ||
      max(abs(rowSums(fitted_values) - 1)) > 1e-5) {
    stop("Python returned invalid generalized propensity scores.")
  }
  observed_p <- fitted_values[cbind(seq_len(nrow(Z)), Tr_code + 1L)]
  weights <- 1 / observed_p

  lsd_by_treatment <- .m_lbcnet_lsd_table(
    result$lsd_by_treatment, treatment_levels
  )
  lsd_values <- if (is.null(result$lsd_values)) NULL else
    as.array(result$lsd_values)
  if (!is.null(lsd_values) &&
      identical(dim(lsd_values), c(n_treatments, K, ncol(Z)))) {
    dimnames(lsd_values) <- list(
      treatment = treatment_names,
      center = format(ck, trim = TRUE),
      covariate = colnames(Z)
    )
  }

  out <- list(
    fitted.values = fitted_values,
    weights = as.numeric(weights),
    loss = as.numeric(result$total_loss),
    lsd_train = list(
      lsd_max = as.numeric(result$max_lsd),
      lsd_mean = as.numeric(result$mean_lsd),
      lsd_by_treatment = lsd_by_treatment
    ),
    treatment_levels = treatment_levels,
    n_treatments = n_treatments,
    parameters = list(
      hidden_dim = as.integer(hidden_dim),
      num_hidden_layers = as.integer(num_hidden_layers),
      vae_epochs = as.integer(vae_epochs),
      vae_lr = as.numeric(vae_lr),
      lr = as.numeric(lr),
      weight_decay = as.numeric(weight_decay),
      balance_lambda = as.numeric(balance_lambda),
      alpha = as.numeric(alpha),
      epsilon = as.numeric(epsilon),
      compute_variance = isTRUE(compute_variance)
    ),
    stopping_criteria = list(
      lsd_threshold = as.numeric(lsd_threshold),
      rolling_window = as.integer(rolling_window),
      max_epochs = as.integer(max_epochs),
      epochs_run = as.integer(result$epochs_run),
      early_stopping = isTRUE(result$early_stopping)
    ),
    estimand = "ATE",
    seed = as.integer(seed),
    call = matched_call,
    formula = formula,
    Z = Z,
    Tr = Tr,
    Tr_code = as.integer(Tr_code),
    ck = as.numeric(ck),
    h = h,
    rho = as.numeric(rho),
    kernel = kernel,
    K = K,
    ps_preliminary = ps_preliminary,
    bandwidth_pilot_method = bandwidth_pilot_method,
    lsd_values = lsd_values,
    tensor_shapes = result$tensor_shapes,
    inference_diagnostics = result$inference_diagnostics
  )

  if (!is.null(Y)) {
    estimates <- as.numeric(result$means)
    se <- as.numeric(result$se_means)
    lower <- as.numeric(result$ci_lower_means)
    upper <- as.numeric(result$ci_upper_means)
    out$Y <- as.numeric(Y)
    out$means <- data.frame(
      treatment = treatment_levels, estimate = estimates, se = se,
      ci_lower = lower, ci_upper = upper,
      check.names = FALSE, stringsAsFactors = FALSE
    )
    covariance <- as.matrix(result$covariance_means)
    storage.mode(covariance) <- "double"
    dimnames(covariance) <- list(treatment_names, treatment_names)
    out$covariance <- covariance
    out$pairwise_ate <- .m_lbcnet_pairwise_table(
      result$pairwise_ate, treatment_levels
    )
    out$influence_functions <- if (is.null(result$influence_functions)) {
      NULL
    } else {
      influence <- as.matrix(result$influence_functions)
      storage.mode(influence) <- "double"
      colnames(influence) <- treatment_names
      influence
    }
  }

  class(out) <- "m_lbcnet"
  if (!is.null(Y) && isTRUE(compute_variance)) {
    out <- .attach_ate_hypothesis_test(out)
  }
  out
}


.m_lbcnet_prepare_inputs <- function(data, formula, Z, Tr, Y, na.action) {
  if (!is.null(formula) && !is.null(data)) {
    data <- as.data.frame(data)
    mf <- stats::model.frame(formula, data = data, na.action = na.action)
    Tr <- stats::model.response(mf)
    Z <- stats::model.matrix(attr(mf, "terms"), mf)
    if ("(Intercept)" %in% colnames(Z)) {
      Z <- Z[, colnames(Z) != "(Intercept)", drop = FALSE]
    }
  }
  if (is.null(Z) || is.null(Tr)) {
    stop("Supply either 'data' and 'formula', or direct 'Z' and 'Tr'.")
  }
  if (is.vector(Z) || (is.data.frame(Z) && ncol(Z) == 1L)) {
    Z <- matrix(Z, ncol = 1L)
  } else if (is.data.frame(Z) || is.matrix(Z)) {
    Z <- as.matrix(Z)
  } else {
    stop("'Z' must be a numeric vector, matrix, or data frame.")
  }
  storage.mode(Z) <- "double"
  if (length(Tr) != nrow(Z)) {
    stop("'Tr' must have one value per row of 'Z'.")
  }
  if (!is.null(Y) && (!is.numeric(Y) || length(Y) != nrow(Z))) {
    stop("'Y' must be numeric with one value per row of 'Z'.")
  }

  working <- as.data.frame(Z, check.names = FALSE)
  working$.m_lbcnet_row_id <- seq_len(nrow(Z))
  working$.m_lbcnet_treatment <- Tr
  if (!is.null(Y)) working$.m_lbcnet_outcome <- Y
  retained <- na.action(working)
  retained_index <- as.integer(retained$.m_lbcnet_row_id)
  Z <- Z[retained_index, , drop = FALSE]
  Tr <- Tr[retained_index]
  if (!is.null(Y)) Y <- Y[retained_index]
  if (is.null(colnames(Z))) colnames(Z) <- paste0("V", seq_len(ncol(Z)))
  colnames(Z) <- make.unique(colnames(Z))
  list(Z = Z, Tr = Tr, Y = Y)
}


.m_lbcnet_treatment_coding <- function(Tr) {
  if (anyNA(Tr)) stop("'Tr' contains missing values after 'na.action'.")
  if (is.factor(Tr)) {
    present <- unique(as.character(Tr))
    levels_out <- levels(Tr)[levels(Tr) %in% present]
    code <- match(as.character(Tr), levels_out) - 1L
  } else if (is.character(Tr)) {
    levels_out <- sort(unique(Tr), method = "radix")
    code <- match(Tr, levels_out) - 1L
  } else if (is.numeric(Tr)) {
    levels_out <- sort(unique(Tr), na.last = NA)
    code <- match(Tr, levels_out) - 1L
  } else {
    stop("'Tr' must be factor, character, integer, or numeric.")
  }
  if (length(levels_out) < 2L) {
    stop("At least two observed treatment groups are required.")
  }
  if (anyNA(code)) stop("Internal treatment coding failed.")
  list(levels = levels_out, code = as.integer(code))
}


.m_lbcnet_bandwidth_pilot <- function(
    Z, Tr_code, treatment_names,
    multinomial_fitter = .m_lbcnet_fit_multinomial_pilot) {
  failure_reason <- NULL
  probabilities <- tryCatch(
    multinomial_fitter(Z, Tr_code, treatment_names),
    error = function(error) {
      failure_reason <<- conditionMessage(error)
      NULL
    }
  )
  if (!is.null(probabilities)) {
    return(list(
      probabilities = probabilities,
      method = "multinomial"
    ))
  }

  warning(
    "Joint multinomial bandwidth pilot failed; using separate ",
    "treatment-versus-rest binary logistic regressions. Reason: ",
    failure_reason,
    call. = FALSE
  )
  list(
    probabilities = .m_lbcnet_fit_treatment_vs_rest_pilot(
      Z, Tr_code, treatment_names
    ),
    method = "treatment_vs_rest"
  )
}


.m_lbcnet_fit_multinomial_pilot <- function(
    Z, Tr_code, treatment_names, maxit = 1000L) {
  n_observations <- nrow(Z)
  n_treatments <- length(treatment_names)
  expected_levels <- as.character(seq_len(n_treatments) - 1L)
  pilot_data <- as.data.frame(Z, check.names = FALSE)
  response_column <- .m_lbcnet_unique_name(
    ".m_lbcnet_response", names(pilot_data)
  )
  pilot_data[[response_column]] <- factor(
    Tr_code, levels = seq_len(n_treatments) - 1L
  )
  pilot_formula <- stats::reformulate(".", response = response_column)

  pilot_warnings <- character()
  maximum_weights <- max(1000, (ncol(Z) + 2L) * n_treatments)
  pilot_result <- withCallingHandlers(
    {
      fit <- nnet::multinom(
        pilot_formula, data = pilot_data, trace = FALSE,
        maxit = maxit, MaxNWts = maximum_weights
      )
      if (length(fit$convergence) != 1L || is.na(fit$convergence) ||
          fit$convergence != 0L) {
        stop("the optimizer did not converge")
      }
      coefficients <- stats::coef(fit)
      if (length(fit$value) != 1L || !is.finite(fit$value) ||
          any(!is.finite(coefficients))) {
        stop("the fitted objective or coefficients were not finite")
      }
      probabilities <- stats::predict(
        fit, newdata = pilot_data, type = "probs"
      )
      list(fit = fit, probabilities = probabilities)
    },
    warning = function(warning_condition) {
      pilot_warnings <<- c(
        pilot_warnings, conditionMessage(warning_condition)
      )
      invokeRestart("muffleWarning")
    }
  )
  if (length(pilot_warnings)) {
    stop(
      "the fit issued warning(s): ",
      paste(unique(pilot_warnings), collapse = "; ")
    )
  }

  probabilities <- pilot_result$probabilities
  if (n_treatments == 2L && is.null(dim(probabilities))) {
    second_probability <- as.numeric(probabilities)
    probabilities <- cbind(1 - second_probability, second_probability)
    colnames(probabilities) <- expected_levels
  } else {
    probabilities <- as.matrix(probabilities)
    if (!is.null(colnames(probabilities))) {
      if (!all(expected_levels %in% colnames(probabilities))) {
        stop("the predicted treatment columns were incomplete")
      }
      probabilities <- probabilities[, expected_levels, drop = FALSE]
    }
  }
  if (!identical(dim(probabilities), c(n_observations, n_treatments)) ||
      any(!is.finite(probabilities)) ||
      any(probabilities < 0 | probabilities > 1) ||
      max(abs(rowSums(probabilities) - 1)) > 1e-7) {
    stop("the predicted probabilities were not a finite N by L simplex")
  }

  probabilities <- pmin(
    pmax(probabilities, .Machine$double.eps),
    1 - .Machine$double.eps
  )
  probabilities <- probabilities / rowSums(probabilities)
  dimnames(probabilities) <- list(NULL, treatment_names)
  probabilities
}


.m_lbcnet_fit_treatment_vs_rest_pilot <- function(
    Z, Tr_code, treatment_names) {
  n_treatments <- length(treatment_names)
  probabilities <- matrix(
    NA_real_, nrow(Z), n_treatments,
    dimnames = list(NULL, treatment_names)
  )
  pilot_data <- as.data.frame(Z, check.names = FALSE)
  response_column <- .m_lbcnet_unique_name(
    ".m_lbcnet_response", names(pilot_data)
  )
  pilot_formula <- stats::reformulate(".", response = response_column)

  for (treatment_index in seq_len(n_treatments)) {
    pilot_data[[response_column]] <- as.integer(
      Tr_code == treatment_index - 1L
    )
    preliminary_fit <- tryCatch(
      suppressWarnings(stats::glm(
        pilot_formula, data = pilot_data, family = "binomial"
      )),
      error = function(error) {
        stop(
          "Treatment-versus-rest bandwidth pilot failed for treatment '",
          treatment_names[treatment_index], "': ",
          conditionMessage(error),
          call. = FALSE
        )
      }
    )
    if (!isTRUE(preliminary_fit$converged)) {
      stop(
        "Treatment-versus-rest bandwidth pilot did not converge for ",
        "treatment '", treatment_names[treatment_index], "'.",
        call. = FALSE
      )
    }
    preliminary_probability <- as.numeric(preliminary_fit$fitted.values)
    if (length(preliminary_probability) != nrow(Z) ||
        any(!is.finite(preliminary_probability))) {
      stop(
        "Treatment-versus-rest bandwidth pilot returned invalid ",
        "probabilities for treatment '", treatment_names[treatment_index],
        "'.",
        call. = FALSE
      )
    }
    probabilities[, treatment_index] <- pmin(
      pmax(preliminary_probability, .Machine$double.eps),
      1 - .Machine$double.eps
    )
  }
  probabilities
}


.m_lbcnet_unique_name <- function(candidate, existing) {
  while (candidate %in% existing) candidate <- paste0(candidate, "_")
  candidate
}


.m_lbcnet_lsd_table <- function(raw, treatment_levels) {
  values <- do.call(rbind, lapply(raw, function(item) {
    c(max_lsd = as.numeric(item$max_lsd),
      mean_lsd = as.numeric(item$mean_lsd))
  }))
  data.frame(
    treatment = treatment_levels,
    max_lsd = values[, "max_lsd"],
    mean_lsd = values[, "mean_lsd"],
    check.names = FALSE, stringsAsFactors = FALSE
  )
}


.m_lbcnet_pairwise_table <- function(raw, treatment_levels) {
  rows <- lapply(raw, function(item) {
    index_1 <- as.integer(item$treatment_1) + 1L
    index_2 <- as.integer(item$treatment_2) + 1L
    data.frame(
      treatment_1 = treatment_levels[index_1],
      treatment_2 = treatment_levels[index_2],
      estimate = as.numeric(item$estimate),
      se = as.numeric(item$se),
      ci_lower = as.numeric(item$ci_lower),
      ci_upper = as.numeric(item$ci_upper),
      check.names = FALSE, stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}
