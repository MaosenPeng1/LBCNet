#' Estimate Survival Difference Using LBC-Net and IPW Nelson–Aalen
#'
#' @description
#' `lbc_net_surv()` extends \code{\link{lbc_net}} to time-to-event outcomes.
#' It first estimates propensity scores via LBC-Net and then combines them
#' with inverse probability weights (IPW) to construct IPW-weighted
#' Nelson–Aalen estimators for the marginal survival functions under treatment
#' and control. The primary estimand is the survival difference
#' \eqn{S_1(t) - S_0(t)} at one or more time points, where the survival
#' estimators are based on the IPW-weighted Nelson–Aalen approach of
#' Deng and Wang (2025).
#'
#' @details
#' Let \eqn{T} be the event or censoring time, \eqn{\Delta} the event
#' indicator (1 = event, 0 = censored), and \eqn{A \in \{0,1\}} the treatment.
#' LBC-Net first estimates the propensity score
#' \eqn{e(X) = P(A = 1 \mid X)} using the same deep learning framework
#' as \code{\link{lbc_net}}, with local balance and calibration constraints.
#'
#' The resulting propensity scores \eqn{\hat e(X)} are used to form
#' Inverse Probability Weights
#' \deqn{
#'   W_i =
#'   \frac{1}{A_i \hat e(X_i) + (1 - A_i)\{1 - \hat e(X_i)\}},
#' }
#' corresponding to the frequency weight function \eqn{\omega^{*}(p) = 1}
#' (no additional tilting). Using these weights, Deng and Wang's (2025)
#' IPW–Nelson–Aalen estimator is applied separately for the treated
#' and control groups to obtain cumulative hazards \eqn{\hat\Lambda_1(t)},
#' \eqn{\hat\Lambda_0(t)}, and survival functions
#' \deqn{
#'   \hat S_a(t) = \exp\{-\hat\Lambda_a(t)\}, \quad a \in \{0,1\}.
#' }
#'
#' The function returns the estimated survival difference
#' \deqn{
#'   \hat\Delta(t) = \hat S_1(t) - \hat S_0(t)
#' }
#' evaluated at a user-specified grid of time points `t_grid`. If the user
#' does not provide a grid or a specific `t_star`, the default evaluation
#' time is set to the median survival time of the control arm, estimated
#' from a standard Kaplan–Meier curve based on observed data.
#'
#' At this stage, `lbc_net_surv()` provides point estimates
#' of \eqn{\hat S_1(t)}, \eqn{\hat S_0(t)}, and their difference
#' \eqn{\hat\Delta(t)} along with variance estimation and
#' influence-function-based standard errors for the survival difference
#' at each evaluation time.
#'
#' @note
#' Compared with \code{\link{lbc_net}}, this function:
#' \itemize{
#'   \item uses \code{time} and \code{delta} instead of an outcome \code{Y},
#'   \item ignores outcome-related arguments such as \code{Y},
#'         \code{outcome_type}, and estimand flags (ATE/ATT/Y).
#' }
#'
#' @inheritParams lbc_net
#'
#' @param time Event or censoring time. This can be:
#'   \itemize{
#'     \item a numeric vector of length equal to the sample size, or
#'     \item a character string naming the column in `data` that contains
#'       the event/censoring times.
#'   }
#'
#' @param delta Event indicator (1 = event, 0 = censored). This can be:
#'   \itemize{
#'     \item a numeric or integer vector of length equal to the sample size, or
#'     \item a character string naming the column in `data` that contains the
#'       event indicator.
#'   }
#'
#' @param t_star Optional numeric value giving a single time point \eqn{t^*}
#'   at which to highlight the survival difference \eqn{S_1(t^*) - S_0(t^*)}
#'   in the output. If `t_star = NULL` (the default), the function sets
#'   \eqn{t^*} to the median survival time of the control group based on
#'   the Kaplan–Meier estimator; if that median is undefined (e.g.,
#'   no events in control), the pooled-sample median is used instead.
#'
#' @param t_grid Optional numeric vector of time points at which to evaluate
#'   the survival curves and survival difference. If `t_grid` is `NULL` and
#'   `t_star` is specified or imputed (default), the function uses a
#'   single-point grid equal to `t_star`. If `t_grid` is non-`NULL`, it
#'   overrides `t_star` for evaluation purposes while `t_star` is still
#'   recorded in the output object.
#'
#' @return
#' An object of class \code{"lbc_net_surv"} (and \code{"lbc_net"}), containing:
#' \itemize{
#'   \item \code{fitted.values}: Estimated propensity scores \eqn{\hat e(X)}.
#'   \item \code{weights}: IPW weights
#'   \item \code{survival}: A list with components:
#'     \itemize{
#'       \item \code{times}: Time grid at which survival functions are evaluated.
#'       \item \code{S1}: Estimated survival function under treatment.
#'       \item \code{S0}: Estimated survival function under control.
#'       \item \code{diff}: Estimated survival difference \code{S1 - S0} at each time.
#'       \item \code{se_diff}: Standard error of the survival difference.
#'       \item \code{ci_lower}: Lower bound of the 95\% Wald CI for the
#'         survival difference.
#'       \item \code{ci_upper}: Upper bound of the 95\% Wald CI for the
#'         survival difference.
#'       \item \code{t_star}: The highlighted time point (median control survival
#'         by default, or user-supplied).
#'     }
#' }
#'
#' @seealso
#' \code{\link{lbc_net}} for propensity score estimation in non-survival settings.
#'
#' @examples
#' \dontrun{
#'   set.seed(123)
#'   n  <- 1000
#'   X1 <- rnorm(n)
#'   X2 <- rnorm(n)
#'   Tr <- rbinom(n, 1, plogis(0.5 * X1 - 0.3 * X2))
#'
#'   lambda0 <- 0.02
#'   hr      <- 0.7
#'   rate    <- lambda0 * ifelse(Tr == 1, hr, 1)
#'   T_true  <- rexp(n, rate = rate)
#'   C       <- rexp(n, rate = 0.01)
#'
#'   time  <- pmin(T_true, C)
#'   delta <- as.integer(T_true <= C)
#'
#'   dat <- data.frame(Tr = Tr, X1 = X1, X2 = X2,
#'                     time = time, delta = delta)
#'
#'   fit_surv <- lbc_net_surv(
#'     data    = dat,
#'     formula = Tr ~ X1 + X2,
#'     time    = "time",
#'     delta   = "delta"
#'   )
#'
#'   # Estimated survival difference and SE at evaluation times
#'   head(cbind(
#'     time = fit_surv$survival$times,
#'     diff = fit_surv$survival$diff,
#'     se   = fit_surv$survival$se_diff
#'   ))
#' }
#'
#' @importFrom stats na.fail model.frame model.response model.matrix glm qnorm
#' @importFrom survival Surv survfit
#' @importFrom reticulate import_from_path
#' @export
lbc_net_surv <- function(data = NULL, formula = NULL,
                         Z = NULL, Tr = NULL,
                         time = NULL, delta = NULL,
                         K = 99, rho = 0.15,
                         na.action = na.fail,
                         gpu = 0, show_progress = TRUE,
                         t_star = NULL, t_grid = NULL,
                         ...,
                         setup_lbcnet_args = list()) {
  
  ## 1. Ensure Python is configured (same pattern as lbc_net)
  if (!reticulate::py_available(initialize = FALSE)) {
    message("Python environment is not set up. Running `setup_lbcnet()`...")
    do.call(setup_lbcnet, setup_lbcnet_args)
  } else {
    message("Python is already set up. Skipping `setup_lbcnet()`.")
  }
  
  ## 2. Extract additional arguments from ...
  args <- list(...)
  
  # Kernel and grid parameters
  ate_flag <- 1
  ck     <- if (!is.null(args$ck)) args$ck else NULL
  h      <- if ("h" %in% names(args)) args$h else NULL
  kernel <- if (!is.null(args$kernel)) args$kernel else "gaussian"
  
  # NN tuning parameters
  seed              <- if (!is.null(args$seed)) args$seed else 100
  hidden_dim        <- if (!is.null(args$hidden_dim)) args$hidden_dim else 100
  num_hidden_layers <- if (!is.null(args$num_hidden_layers)) args$num_hidden_layers else 1
  vae_epochs        <- if (!is.null(args$vae_epochs)) args$vae_epochs else 250
  vae_lr            <- if (!is.null(args$vae_lr)) args$vae_lr else 0.01
  max_epochs        <- if (!is.null(args$max_epochs)) args$max_epochs else 5000
  lr                <- if (!is.null(args$lr)) args$lr else 0.05
  weight_decay      <- if (!is.null(args$weight_decay)) args$weight_decay else 1e-5
  balance_lambda    <- if (!is.null(args$balance_lambda)) args$balance_lambda else 1.0
  epsilon           <- if (!is.null(args$epsilon)) args$epsilon else 0.001
  lsd_threshold     <- if (!is.null(args$lsd_threshold)) args$lsd_threshold else 2
  rolling_window    <- if (!is.null(args$rolling_window)) args$rolling_window else 5
  
  ## 3. Load Python module for survival
  pymod <- reticulate::import_from_path(
    "lbc_net_surv",
    path = system.file("python", package = "LBCNet")
  )
  
  ## 4. Handle formula-based input for Z and Tr (as in lbc_net)
  if (!is.null(formula) && !is.null(data)) {
    model_frame <- model.frame(formula, data, na.action = na.action)
    Tr <- model.response(model_frame)
    Z  <- model.matrix(attr(model_frame, "terms"), model_frame)
    
    # Drop intercept if exists
    if ("(Intercept)" %in% colnames(Z)) {
      Z <- Z[, -1, drop = FALSE]
    }
  }
  
  ## 5. Resolve time and delta from data if given as column names
  if (!is.null(data)) {
    if (is.character(time) && length(time) == 1L) {
      if (!time %in% names(data)) {
        stop("Column '", time, "' not found in 'data'.")
      }
      time_vec <- data[[time]]
    } else {
      time_vec <- time
    }
    
    if (is.character(delta) && length(delta) == 1L) {
      if (!delta %in% names(data)) {
        stop("Column '", delta, "' not found in 'data'.")
      }
      delta_vec <- data[[delta]]
    } else {
      delta_vec <- delta
    }
  } else {
    time_vec  <- time
    delta_vec <- delta
  }
  
  ## 6. Basic checks for Z, Tr, time, delta
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
  
  if (!is.numeric(Tr) || length(Tr) != nrow(Z)) {
    stop("Tr must be a numeric vector with the same length as number of rows in Z.")
  }
  
  if (is.null(time_vec) || is.null(delta_vec)) {
    stop("Both 'time' and 'delta' must be provided.")
  }
  
  if (length(time_vec) != length(Tr) || length(delta_vec) != length(Tr)) {
    stop("'time', 'delta', and 'Tr' must have the same length.")
  }
  
  # ---- Outcome leakage detection for survival (time, delta) ----
  # Assumes `data`, `time_vec`, and `delta_vec` are defined
  
  bad_cols <- character(0)
  
  # Check which column(s) in `data` are identical to the time variable
  matching_time <- vapply(
    data,
    function(col) isTRUE(all.equal(col, time_vec)),
    logical(1)
  )
  time_cols_in_data <- names(data)[matching_time]
  bad_cols <- c(bad_cols, time_cols_in_data)
  
  # Check which column(s) in `data` are identical to the delta variable
  matching_delta <- vapply(
    data,
    function(col) isTRUE(all.equal(col, delta_vec)),
    logical(1)
  )
  delta_cols_in_data <- names(data)[matching_delta]
  bad_cols <- c(bad_cols, delta_cols_in_data)
  
  bad_cols <- unique(bad_cols)
  
  if (length(bad_cols) > 0L) {
    warning(
      sprintf(
        "The survival outcome variables (time and/or delta) appear to be data column(s): %s.\n",
        paste(bad_cols, collapse = ", ")
      ),
      "Including time or delta in the formula (e.g., Tr ~ .) causes outcome leakage\n",
      "and invalidates the propensity model.\n",
      "Use a formula such as:  Tr ~ . - time - delta\n",
      call. = FALSE
    )
  }
  
  ## 7. Apply NA action jointly to (Z, Tr, time, delta)
  tmp_all <- data.frame(Z, Tr = Tr, time = time_vec, delta = delta_vec)
  na_res  <- na.action(tmp_all)
  
  Z         <- as.matrix(na_res[, seq_len(ncol(Z)), drop = FALSE])
  Tr        <- na_res[["Tr"]]
  time_vec  <- na_res[["time"]]
  delta_vec <- na_res[["delta"]]
  
  Tr        <- as.numeric(Tr)
  time_vec  <- as.numeric(time_vec)
  delta_vec <- as.numeric(delta_vec)
  
  ## 8. Calculate logistic PS for ck/h calculation (as in lbc_net)
  message("Calculating propensity scores for ck/h calculation...")
  log.fit <- glm(Tr ~ ., data = as.data.frame(Z), family = "binomial")
  ps_log  <- log.fit$fitted.values
  
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
    colnames(Z) <- paste0("V", seq_len(ncol(Z)))
  }
  
  ## 9. Default t_star: median survival of control arm (KM)
  if (is.null(t_star)) {
    # KM for A = 0
    km_ctrl <- survival::survfit(
      survival::Surv(time_vec, delta_vec) ~ 1,
      subset = (Tr == 0)
    )
    
    time_vals <- km_ctrl$time
    surv_vals <- km_ctrl$surv
    
    # If no control observations OR no time points, fall back to pooled KM
    if (length(time_vals) == 0L || length(surv_vals) == 0L) {
      km_all <- survival::survfit(
        survival::Surv(time_vec, delta_vec) ~ 1
      )
      time_vals <- km_all$time
      surv_vals <- km_all$surv
    }
    
    # If still no time points, we cannot define a median
    if (length(time_vals) == 0L || length(surv_vals) == 0L) {
      stop(
        "Cannot compute default t_star: no events or time information ",
        "available in either treatment group."
      )
    }
    
    # Find first time where survival <= 0.5
    idx <- which(surv_vals <= 0.5)[1]
    
    if (is.na(idx)) {
      # Survival never crosses 0.5 → median not reached
      # Use largest observed time as default t_star
      t_med <- max(time_vals, na.rm = TRUE)
    } else {
      t_med <- time_vals[idx]
    }
    
    t_star <- t_med
  }
    
  # If no explicit grid is given, use t_star only
  if (is.null(t_grid)) {
    t_grid <- as.numeric(t_star)
  } else {
    t_grid <- as.numeric(t_grid)
  }
  
  ## 10. Prepare data frame for Python
  data_df <- as.data.frame(cbind(Tr, time_vec, delta_vec, Z))
  colnames(data_df)[1:3] <- c("Tr", "time", "delta")
  
  ## 11. Call Python function implementing Deng–Wang IPW NA estimator
  result <- pymod$run_lbc_net_surv(
    data_df        = data_df,
    Z_columns      = colnames(Z),
    T_column       = "Tr",
    time_column    = "time",
    delta_column   = "delta",
    ck             = ck,
    h              = h,
    kernel         = kernel,
    gpu            = as.integer(gpu),
    seed           = as.integer(seed),
    hidden_dim     = as.integer(hidden_dim),
    L              = as.integer(num_hidden_layers + 1),
    vae_epochs     = as.integer(vae_epochs),
    vae_lr         = vae_lr,
    max_epochs     = as.integer(max_epochs),
    lr             = lr,
    weight_decay   = weight_decay,
    balance_lambda = balance_lambda,
    epsilon        = epsilon,
    lsd_threshold  = lsd_threshold,
    rolling_window = as.integer(rolling_window),
    show_progress  = show_progress,
    t_grid         = t_grid
  )
  
  ## 12. Unpack Python results
  propensity_scores <- result$propensity_scores
  total_loss        <- result$total_loss
  lsd_max           <- result$max_lsd
  lsd_mean          <- result$mean_lsd
  
  surv_times <- result$times
  S1         <- result$S1
  S0         <- result$S0
  surv_diff  <- result$surv_diff
  se_diff  <- result$se
  ci_lower <- result$ci_lower
  ci_upper <- result$ci_upper
  
  ## 13. Construct ATE-type IPW weights (ω*(p) = 1)
  ipw <- 1 / (Tr * propensity_scores + (1 - Tr) * (1 - propensity_scores))
  
  ## 14. Assemble output object
  out <- list(
    fitted.values = propensity_scores,
    weights       = ipw,
    survival      = list(
      times    = as.numeric(surv_times),
      S1       = as.numeric(S1),
      S0       = as.numeric(S0),
      diff     = as.numeric(surv_diff),
      se_diff  = as.numeric(se_diff),
      ci_lower = as.numeric(ci_lower),
      ci_upper = as.numeric(ci_upper)
    ),
    loss = total_loss,
    lsd_train = list(
      lsd_max  = lsd_max,
      lsd_mean = lsd_mean
    ),
    parameters = list(
      hidden_dim        = hidden_dim,
      num_hidden_layers = num_hidden_layers,
      vae_lr            = vae_lr,
      lr                = lr,
      weight_decay      = weight_decay,
      balance_lambda    = balance_lambda,
      epsilon           = epsilon
    ),
    stopping_criteria = list(
      lsd_threshold  = lsd_threshold,
      rolling_window = rolling_window,
      max_epochs     = max_epochs
    ),
    seed        = seed,
    call        = match.call(),
    formula     = formula,
    Z           = Z,
    Tr          = Tr,
    time        = time_vec,
    delta       = delta_vec,
    ck          = ck,
    h           = h,
    rho         = rho,
    kernel      = kernel,
    K           = K,
    ps_logistic = ps_log,
    ate_flag    = ate_flag
  )
  
  class(out) <- c("lbc_net_surv", "lbc_net")
  return(out)
}
