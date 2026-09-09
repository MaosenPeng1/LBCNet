m_lbcnet_python_ready <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(FALSE)
  available <- tryCatch(
    reticulate::py_available(initialize = TRUE),
    error = function(e) FALSE
  )
  if (!isTRUE(available)) return(FALSE)
  all(vapply(
    c("torch", "numpy", "pandas", "tqdm"),
    function(module) {
      tryCatch(reticulate::py_module_available(module),
               error = function(e) FALSE)
    },
    logical(1)
  ))
}

skip_if_no_m_lbcnet_python <- function() {
  testthat::skip_if_not(
    m_lbcnet_python_ready(),
    "A local Python with torch, numpy, pandas, and tqdm is required."
  )
}

m_lbcnet_test_data <- local({
  set.seed(824)
  n <- 36
  Z <- matrix(
    rnorm(n * 2), nrow = n, ncol = 2,
    dimnames = list(NULL, c("x1", "x2"))
  )
  Tr <- factor(
    rep(c("A", "B", "C"), length.out = n),
    levels = c("A", "B", "C")
  )
  Y <- 2 + as.numeric(Tr) + 0.5 * Z[, 1] - Z[, 2] + rnorm(n)
  list(
    Z = Z,
    Tr = Tr,
    Y = Y,
    data = data.frame(Tr = Tr, x1 = Z[, 1], x2 = Z[, 2])
  )
})

m_lbcnet_fit_cache <- new.env(parent = emptyenv())

get_m_lbcnet_test_fit <- function(which = c("gps", "outcome")) {
  which <- match.arg(which)
  if (exists(which, envir = m_lbcnet_fit_cache, inherits = FALSE)) {
    return(get(which, envir = m_lbcnet_fit_cache, inherits = FALSE))
  }
  skip_if_no_m_lbcnet_python()
  d <- m_lbcnet_test_data
  if (which == "gps") {
    fit <- m_lbcnet(
      Z = d$Z, Tr = d$Tr,
      h = rep(0.25, 19),
      hidden_dim = 3, num_hidden_layers = 0,
      vae_epochs = 0, max_epochs = 1,
      compute_variance = FALSE, show_progress = FALSE, seed = 91
    )
  } else {
    fit <- m_lbcnet(
      Z = d$Z, Tr = d$Tr, Y = d$Y,
      ck = 0.5, h = matrix(0.3, nrow = 3, ncol = 1),
      hidden_dim = 2, num_hidden_layers = 0,
      vae_epochs = 0, max_epochs = 1,
      show_progress = FALSE, seed = 92
    )
  }
  assign(which, fit, envir = m_lbcnet_fit_cache)
  fit
}


test_that("M-LBCNet uses the binary LBC-Net Python setup mechanism", {
  implementation <- paste(
    deparse(body(m_lbcnet), width.cutoff = 500L), collapse = "\n"
  )

  expect_match(
    implementation,
    "reticulate::py_available(initialize = FALSE)",
    fixed = TRUE
  )
  expect_false(grepl(
    "reticulate::py_available(initialize = TRUE)",
    implementation,
    fixed = TRUE
  ))
  expect_match(
    implementation,
    "do.call(setup_lbcnet, setup_lbcnet_args)",
    fixed = TRUE
  )
  expect_match(
    implementation,
    paste0(
      "reticulate::import_from_path(\"m_lbcnet\", path = ",
      "system.file(\"python\", package = \"LBCNet\")"
    ),
    fixed = TRUE
  )
  expect_false(grepl("result$weights", implementation, fixed = TRUE))
  expect_false(grepl("__file__", implementation, fixed = TRUE))
  expect_false(grepl(
    "covariance - t(covariance)", implementation, fixed = TRUE
  ))
  expect_false(grepl("dim(influence)", implementation, fixed = TRUE))
  expect_false(exists(
    ".m_lbcnet_initialize_python",
    envir = asNamespace("LBCNet"), inherits = FALSE
  ))
})


test_that("M-LBCNet configures Python before importing its local module", {
  state <- new.env(parent = emptyenv())
  state$setup_calls <- 0L
  state$messages <- character()

  testthat::local_mocked_bindings(
    py_available = function(initialize) {
      state$initialize <- initialize
      FALSE
    },
    import_from_path = function(module, path, ...) {
      state$module <- module
      state$path <- path
      stop("mock import reached", call. = FALSE)
    },
    .package = "reticulate"
  )
  testthat::local_mocked_bindings(
    setup_lbcnet = function(...) {
      state$setup_calls <- state$setup_calls + 1L
      state$setup_args <- list(...)
    },
    .package = "LBCNet"
  )

  setup_args <- list(envname = "mock-lbcnet", python_version = "3.10")
  error <- tryCatch(
    withCallingHandlers(
      m_lbcnet(
        Z = m_lbcnet_test_data$Z,
        Tr = m_lbcnet_test_data$Tr,
        ck = 0.5, h = 0.3,
        compute_variance = FALSE, show_progress = FALSE,
        setup_lbcnet_args = setup_args
      ),
      message = function(condition) {
        state$messages <- c(state$messages, conditionMessage(condition))
        invokeRestart("muffleMessage")
      }
    ),
    error = identity
  )

  expect_s3_class(error, "error")
  expect_identical(conditionMessage(error), "mock import reached")
  expect_identical(state$initialize, FALSE)
  expect_identical(state$setup_calls, 1L)
  expect_identical(state$setup_args, setup_args)
  expect_identical(
    state$messages,
    "Python environment is not set up. Running `setup_lbcnet()`...\n"
  )
  expect_identical(state$module, "m_lbcnet")
  expect_identical(
    state$path,
    system.file("python", package = "LBCNet")
  )
})


test_that("M-LBCNet skips setup when Python is already initialized", {
  state <- new.env(parent = emptyenv())
  state$setup_calls <- 0L
  state$messages <- character()

  testthat::local_mocked_bindings(
    py_available = function(initialize) {
      state$initialize <- initialize
      TRUE
    },
    import_from_path = function(...) {
      stop("mock import reached", call. = FALSE)
    },
    .package = "reticulate"
  )
  testthat::local_mocked_bindings(
    setup_lbcnet = function(...) {
      state$setup_calls <- state$setup_calls + 1L
    },
    .package = "LBCNet"
  )

  error <- tryCatch(
    withCallingHandlers(
      m_lbcnet(
        Z = m_lbcnet_test_data$Z,
        Tr = m_lbcnet_test_data$Tr,
        ck = 0.5, h = 0.3,
        compute_variance = FALSE, show_progress = FALSE
      ),
      message = function(condition) {
        state$messages <- c(state$messages, conditionMessage(condition))
        invokeRestart("muffleMessage")
      }
    ),
    error = identity
  )

  expect_s3_class(error, "error")
  expect_identical(conditionMessage(error), "mock import reached")
  expect_identical(state$initialize, FALSE)
  expect_identical(state$setup_calls, 0L)
  expect_identical(
    state$messages,
    "Python is already set up. Skipping `setup_lbcnet()`.\n"
  )
})


test_that("M-LBCNet progress and completion output parallel binary LBC-Net", {
  skip_if_no_m_lbcnet_python()
  module <- reticulate::import_from_path(
    "m_lbcnet", path = system.file("python", package = "LBCNet"),
    convert = FALSE
  )
  reticulate::py_run_string(paste(
    "class RecordingTqdm:",
    "    instances = []",
    "    def __init__(self, *args, **kwargs):",
    "        self.args = args",
    "        self.kwargs = kwargs",
    "        self.postfixes = []",
    "        self.updates = []",
    "        self.close_calls = 0",
    "        type(self).instances.append(self)",
    "    def set_postfix(self, values):",
    "        self.postfixes.append(dict(values))",
    "    def update(self, value):",
    "        self.updates.append(value)",
    "    def close(self):",
    "        self.close_calls += 1",
    "class FakeTime:",
    "    def __init__(self):",
    "        self.values = iter([100.0, 102.0])",
    "    def time(self):",
    "        return next(self.values)",
    sep = "\n"
  ))

  original_tqdm <- module$tqdm
  original_time <- module$time
  on.exit(
    reticulate::py_set_attr(module, "tqdm", original_tqdm), add = TRUE
  )
  on.exit(
    reticulate::py_set_attr(module, "time", original_time), add = TRUE
  )
  reticulate::py_set_attr(
    module, "tqdm", reticulate::py$RecordingTqdm
  )
  reticulate::py_set_attr(
    module, "time", reticulate::py$FakeTime()
  )

  runner_output <- reticulate::py_capture_output(
    module$run_m_lbcnet(
      data_df = data.frame(
        z = rep(c(-1, 0, 1), 3),
        tr = rep(0:2, each = 3),
        y = seq_len(9)
      ),
      Z_columns = "z", T_column = "tr", Y_column = "y",
      n_treatments = 3L, ck = 0.5, h = matrix(0.3, 3, 1),
      hidden_dim = 2L, num_layers = 1L,
      vae_epochs = 0L, max_epochs = 1L,
      show_progress = TRUE, compute_variance = FALSE
    ),
    type = c("stdout", "stderr")
  )

  instance <- reticulate::py$RecordingTqdm$instances[[1]]
  arguments <- reticulate::py_to_r(instance$kwargs)
  postfixes <- reticulate::py_to_r(instance$postfixes)
  expect_length(reticulate::py_to_r(instance$args), 0L)
  expect_identical(
    names(arguments),
    c("total", "desc", "position", "leave", "bar_format")
  )
  expect_identical(arguments$total, 1L)
  expect_identical(arguments$desc, "Training Progress")
  expect_identical(arguments$position, 0L)
  expect_true(arguments$leave)
  expect_identical(
    arguments$bar_format,
    "{l_bar}{bar} {n_fmt}/{total_fmt} [{rate_fmt} {postfix}]"
  )
  expect_identical(
    names(postfixes[[1]]),
    c("Remaining Time (s)", "Elapsed Time (s)", "Loss")
  )
  expect_identical(postfixes[[1]][["Remaining Time (s)"]], "0.00")
  expect_identical(postfixes[[1]][["Elapsed Time (s)"]], "2.00")
  expect_match(postfixes[[1]]$Loss, "^[0-9]+\\.[0-9]{4}$")
  expect_identical(reticulate::py_to_r(instance$updates), 1L)
  expect_identical(reticulate::py_to_r(instance$close_calls), 1L)

  max_epoch_message <- paste0(
    "\u26a0\ufe0f Stopping criterion not met at max epochs. ",
    "Try increasing `max_epochs` or adjusting `lsd_threshold` ",
    "for better convergence."
  )
  postprocessing_message <- paste0(
    "Starting post-processing: computing treatment-specific means, ",
    "pairwise ATEs, and variance..."
  )
  completion_message <-
    "\u2705 M-LBCNet training completed successfully."
  expect_match(runner_output, max_epoch_message, fixed = TRUE)
  expect_match(runner_output, postprocessing_message, fixed = TRUE)
  expect_match(runner_output, completion_message, fixed = TRUE)
  expect_lt(
    regexpr(max_epoch_message, runner_output, fixed = TRUE)[[1]],
    regexpr(postprocessing_message, runner_output, fixed = TRUE)[[1]]
  )
  expect_lt(
    regexpr(postprocessing_message, runner_output, fixed = TRUE)[[1]],
    regexpr(completion_message, runner_output, fixed = TRUE)[[1]]
  )
})


test_that("M-LBCNet reports its unchanged rolling-LSD early stop", {
  skip_if_no_m_lbcnet_python()
  module <- reticulate::import_from_path(
    "m_lbcnet", path = system.file("python", package = "LBCNet"),
    convert = FALSE
  )
  reticulate::py_run_string(paste(
    "import builtins",
    "def shortcut_range(stop):",
    "    return [199] if stop == 200 else builtins.range(stop)",
    sep = "\n"
  ))
  builtins <- reticulate::import("builtins", convert = FALSE)
  on.exit(
    reticulate::py_set_attr(module, "range", builtins$range), add = TRUE
  )
  reticulate::py_set_attr(
    module, "range", reticulate::py$shortcut_range
  )

  early_result <- NULL
  early_output <- reticulate::py_capture_output(
    early_result <- module$run_m_lbcnet(
      data_df = data.frame(
        z = rep(c(-1, 0, 1), 3),
        tr = rep(0:2, each = 3)
      ),
      Z_columns = "z", T_column = "tr", Y_column = NULL,
      n_treatments = 3L, ck = 0.5, h = matrix(0.3, 3, 1),
      hidden_dim = 2L, num_layers = 1L,
      vae_epochs = 0L, max_epochs = 200L,
      lsd_threshold = 1e100, rolling_window = 1L,
      show_progress = FALSE, compute_variance = FALSE
    ),
    type = "stdout"
  )

  expect_identical(reticulate::py_to_r(early_result$epochs_run), 200L)
  expect_true(reticulate::py_to_r(early_result$early_stopping))
  expect_match(
    early_output,
    paste0(
      "\u2705 Stopping early at epoch 200 ",
      "(rolling average max LSD < 1e+100%)"
    ),
    fixed = TRUE
  )
  expect_false(grepl(
    "Stopping criterion not met", early_output, fixed = TRUE
  ))
})


test_that("basic three-arm fit is a joint simplex GPS with ATE weights", {
  fit <- get_m_lbcnet_test_fit("gps")
  d <- m_lbcnet_test_data

  expect_s3_class(fit, "m_lbcnet")
  expect_false(inherits(fit, "lbc_net"))
  expect_equal(dim(fit$fitted.values), c(nrow(d$Z), 3))
  expect_true(all(is.finite(fit$fitted.values)))
  expect_true(all(fit$fitted.values > 0))
  expect_equal(rowSums(fit$fitted.values), rep(1, nrow(d$Z)),
               tolerance = 1e-6)
  expect_true(all(is.finite(fit$weights) & fit$weights > 0))
  expected_weights <- 1 / fit$fitted.values[
    cbind(seq_len(nrow(d$Z)), fit$Tr_code + 1L)
  ]
  expect_equal(fit$weights, expected_weights, tolerance = 1e-8)
  expect_identical(fit$K, 19L)
  expect_equal(fit$ck, seq_len(19) / 20)
  expect_identical(fit$estimand, "ATE")
  expect_equal(dim(fit$h), c(3, 19))
  expect_null(fit$ps_preliminary)
  expect_identical(fit$bandwidth_pilot_method, "not_used")
  expect_null(fit$Y)
})


test_that("Python moments match the requested raw stack and loss scaling", {
  skip_if_no_m_lbcnet_python()
  python_path <- system.file("python", package = "LBCNet")
  module <- reticulate::import_from_path(
    "m_lbcnet", path = python_path, convert = FALSE
  )
  torch <- reticulate::import("torch", convert = FALSE)

  gps <- rbind(
    c(0.50, 0.30, 0.20),
    c(0.25, 0.50, 0.25),
    c(0.20, 0.30, 0.50),
    c(0.40, 0.35, 0.25)
  )
  treatment <- c(0L, 1L, 2L, 0L)
  Z <- cbind(intercept = 1, x = c(-1, 0, 1, 2))
  ck <- c(0.25, 0.75)
  h <- rbind(c(0.20, 0.30), c(0.25, 0.35), c(0.30, 0.40))
  balance_lambda <- 4

  gps_t <- torch$tensor(gps, dtype = torch$float32)
  treatment_t <- torch$tensor(treatment, dtype = torch$long)
  Z_t <- torch$tensor(Z, dtype = torch$float32)
  ck_t <- torch$tensor(ck, dtype = torch$float32)
  h_t <- torch$tensor(h, dtype = torch$float32)
  observed <- reticulate::py_to_r(
    module$m_lbcnet_moments(
      gps_t, treatment_t, Z_t, ck_t, h_t,
      kernel_id = 0L, balance_lambda = balance_lambda
    )$detach()$cpu()$numpy()
  )

  R <- vapply(0:2, function(index) as.numeric(treatment == index), numeric(4))
  expected_blocks <- list()
  block <- 0L
  for (treatment_index in seq_len(3)) {
    for (grid_index in seq_along(ck)) {
      omega <- stats::dnorm(
        (gps[, treatment_index] - ck[grid_index]) /
          h[treatment_index, grid_index]
      ) / h[treatment_index, grid_index]
      balance <- (
        omega * (R[, treatment_index] / gps[, treatment_index] - 1)
      ) * Z
      calibration <- sqrt(balance_lambda) * omega *
        (R[, treatment_index] - gps[, treatment_index]) /
        (ck[grid_index] * (1 - ck[grid_index]))
      block <- block + 1L
      expected_blocks[[block]] <- cbind(balance, calibration)
    }
  }
  expected <- do.call(cbind, expected_blocks)
  expect_equal(observed, unname(expected), tolerance = 1e-5)

  unscaled <- reticulate::py_to_r(
    module$m_lbcnet_loss(
      gps_t, treatment_t, Z_t, ck_t, h_t,
      kernel_id = 0L, balance_lambda = balance_lambda,
      optimizer_scale = FALSE
    )$detach()$cpu()$item()
  )
  scaled <- reticulate::py_to_r(
    module$m_lbcnet_loss(
      gps_t, treatment_t, Z_t, ck_t, h_t,
      kernel_id = 0L, balance_lambda = balance_lambda,
      optimizer_scale = TRUE
    )$detach()$cpu()$item()
  )
  expected_q <- sum(colMeans(expected)^2) / (3 * length(ck))
  expect_equal(as.numeric(unscaled), expected_q, tolerance = 1e-5)
  expect_equal(as.numeric(scaled / unscaled), nrow(Z)^2, tolerance = 1e-5)
})


test_that("factor labels are retained across every multi-arm result", {
  fit <- get_m_lbcnet_test_fit("outcome")
  expected <- c("A", "B", "C")

  expect_identical(colnames(fit$fitted.values), expected)
  expect_identical(as.character(fit$treatment_levels), expected)
  expect_identical(as.character(fit$means$treatment), expected)
  expect_identical(rownames(fit$covariance), expected)
  expect_identical(colnames(fit$covariance), expected)
  expect_identical(colnames(fit$influence_functions), expected)
  expect_setequal(as.character(fit$pairwise_ate$treatment_1), c("A", "B"))
  expect_setequal(as.character(fit$pairwise_ate$treatment_2), c("B", "C"))

  numeric_coding <- LBCNet:::.m_lbcnet_treatment_coding(
    c(30, 10, 20, 30)
  )
  expect_equal(numeric_coding$levels, c(10, 20, 30))
  character_coding <- LBCNet:::.m_lbcnet_treatment_coding(
    c("z", "a", "m", "z")
  )
  expect_identical(character_coding$levels, c("a", "m", "z"))
})


test_that("formula and direct input are reproducible and equivalent", {
  skip_if_no_m_lbcnet_python()
  d <- m_lbcnet_test_data
  tuning <- list(
    ck = 0.5, h = 0.3, hidden_dim = 2, num_hidden_layers = 0,
    vae_epochs = 0, max_epochs = 1, compute_variance = FALSE,
    show_progress = FALSE, seed = 121
  )
  direct <- do.call(
    m_lbcnet,
    c(list(Z = d$Z, Tr = d$Tr), tuning)
  )
  formula_fit <- do.call(
    m_lbcnet,
    c(list(data = d$data, formula = Tr ~ x1 + x2), tuning)
  )
  direct_again <- do.call(
    m_lbcnet,
    c(list(Z = d$Z, Tr = d$Tr), tuning)
  )

  expect_equal(formula_fit$fitted.values, direct$fitted.values,
               tolerance = 1e-6)
  expect_equal(direct_again$fitted.values, direct$fitted.values,
               tolerance = 1e-6)
  expect_identical(formula_fit$treatment_levels, direct$treatment_levels)
})


test_that("adaptive bandwidth pilot is one joint multinomial GPS", {
  d <- m_lbcnet_test_data
  treatment_names <- as.character(levels(d$Tr))
  treatment_code <- as.integer(d$Tr) - 1L
  pilot <- LBCNet:::.m_lbcnet_bandwidth_pilot(
    d$Z, treatment_code, treatment_names
  )

  pilot_data <- data.frame(
    .treatment = factor(treatment_code, levels = 0:2),
    as.data.frame(d$Z, check.names = FALSE),
    check.names = FALSE
  )
  expected_fit <- nnet::multinom(
    .treatment ~ ., data = pilot_data, trace = FALSE
  )
  expected_probabilities <- stats::predict(
    expected_fit, newdata = pilot_data, type = "probs"
  )

  expect_identical(pilot$method, "multinomial")
  expect_identical(dim(pilot$probabilities), c(nrow(d$Z), 3L))
  expect_identical(colnames(pilot$probabilities), treatment_names)
  expect_equal(
    rowSums(pilot$probabilities), rep(1, nrow(d$Z)), tolerance = 1e-10
  )
  expect_equal(
    unname(pilot$probabilities), unname(expected_probabilities),
    tolerance = 1e-8
  )
})


test_that("multinomial pilot handles two arms and response-name collisions", {
  d <- m_lbcnet_test_data
  keep <- d$Tr != "C"
  Z <- d$Z[keep, , drop = FALSE]
  colnames(Z) <- c(".m_lbcnet_response", "non syntactic name")
  Tr <- droplevels(d$Tr[keep])
  pilot <- LBCNet:::.m_lbcnet_bandwidth_pilot(
    Z, as.integer(Tr) - 1L, as.character(levels(Tr))
  )

  expect_identical(pilot$method, "multinomial")
  expect_identical(dim(pilot$probabilities), c(nrow(Z), 2L))
  expect_identical(colnames(pilot$probabilities), c("A", "B"))
  expect_true(all(is.finite(pilot$probabilities)))
  expect_equal(rowSums(pilot$probabilities), rep(1, nrow(Z)),
               tolerance = 1e-10)
})


test_that("failed multinomial pilot warns and uses treatment-vs-rest fits", {
  d <- m_lbcnet_test_data
  treatment_names <- as.character(levels(d$Tr))
  treatment_code <- as.integer(d$Tr) - 1L
  failing_fitter <- function(...) stop("forced numerical failure")

  fallback <- NULL
  expect_warning(
    fallback <- LBCNet:::.m_lbcnet_bandwidth_pilot(
      d$Z, treatment_code, treatment_names,
      multinomial_fitter = failing_fitter
    ),
    "Joint multinomial bandwidth pilot failed.*treatment-versus-rest"
  )

  expected <- vapply(
    seq_along(treatment_names),
    function(index) {
      stats::glm(
        as.integer(treatment_code == index - 1L) ~ d$Z,
        family = "binomial"
      )$fitted.values
    },
    numeric(nrow(d$Z))
  )
  expect_identical(fallback$method, "treatment_vs_rest")
  expect_identical(dim(fallback$probabilities), c(nrow(d$Z), 3L))
  expect_identical(colnames(fallback$probabilities), treatment_names)
  expect_equal(unname(fallback$probabilities), unname(expected),
               tolerance = 1e-8)
})


test_that("multinomial pilot rejects optimizer nonconvergence", {
  d <- m_lbcnet_test_data
  expect_error(
    LBCNet:::.m_lbcnet_fit_multinomial_pilot(
      d$Z, as.integer(d$Tr) - 1L, as.character(levels(d$Tr)),
      maxit = 1L
    ),
    "optimizer did not converge"
  )
})


test_that("custom grids and bandwidth forms are validated", {
  skip_if_no_m_lbcnet_python()
  d <- m_lbcnet_test_data
  common <- list(
    Z = d$Z, Tr = d$Tr, ck = c(0.25, 0.75),
    hidden_dim = 2, num_hidden_layers = 0, vae_epochs = 0,
    max_epochs = 1, compute_variance = FALSE, show_progress = FALSE
  )
  vector_fit <- do.call(m_lbcnet, c(common, list(h = c(0.2, 0.3))))
  matrix_h <- rbind(c(0.2, 0.3), c(0.25, 0.35), c(0.3, 0.4))
  matrix_fit <- do.call(m_lbcnet, c(common, list(h = matrix_h)))
  automatic_fit <- m_lbcnet(
    Z = d$Z, Tr = d$Tr, K = 1, rho = 0.2,
    hidden_dim = 2, num_hidden_layers = 0, vae_epochs = 0,
    max_epochs = 1, compute_variance = FALSE, show_progress = FALSE
  )
  expected_h <- do.call(rbind, lapply(
    seq_len(3),
    function(index) span_bw(
      0.2, automatic_fit$ck, automatic_fit$ps_preliminary[, index]
    )
  ))

  expect_identical(vector_fit$K, 2L)
  expect_equal(unname(vector_fit$h), matrix(
    rep(c(0.2, 0.3), each = 3), nrow = 3
  ))
  expect_equal(unname(matrix_fit$h), matrix_h)
  expect_null(vector_fit$ps_preliminary)
  expect_null(matrix_fit$ps_preliminary)
  expect_identical(vector_fit$bandwidth_pilot_method, "not_used")
  expect_identical(matrix_fit$bandwidth_pilot_method, "not_used")
  expect_equal(unname(automatic_fit$h), expected_h)
  expect_identical(dim(automatic_fit$h), c(3L, 1L))
  expect_identical(
    dim(automatic_fit$ps_preliminary), c(nrow(d$Z), 3L)
  )
  expect_identical(
    colnames(automatic_fit$ps_preliminary), as.character(levels(d$Tr))
  )
  expect_equal(
    rowSums(automatic_fit$ps_preliminary), rep(1, nrow(d$Z)),
    tolerance = 1e-10
  )
  expect_identical(automatic_fit$bandwidth_pilot_method, "multinomial")
  expect_error(
    do.call(m_lbcnet, c(common, list(h = 0.2))),
    "length K"
  )
  expect_error(
    do.call(m_lbcnet, c(common, list(h = matrix(0.2, 2, 2)))),
    "n_treatments by K"
  )
  expect_error(
    do.call(m_lbcnet, c(common, list(h = c(0.2, -0.1)))),
    "strictly positive"
  )
})


test_that("treatment, missingness, and epsilon validation are explicit", {
  d <- m_lbcnet_test_data
  expect_error(
    m_lbcnet(Z = d$Z, Tr = rep("only", nrow(d$Z))),
    "At least two observed treatment"
  )
  Tr_missing <- d$Tr
  Tr_missing[1] <- NA
  expect_error(
    m_lbcnet(Z = d$Z, Tr = Tr_missing),
    "missing values"
  )
  expect_error(
    m_lbcnet(Z = d$Z, Tr = d$Tr, epsilon = 1 / 3),
    "strictly less than 1"
  )

  skip_if_no_m_lbcnet_python()
  expect_error(
    m_lbcnet(
      Z = d$Z, Tr = d$Tr, ck = 0.5, h = 0.3, epsilon = -0.01,
      hidden_dim = 2, num_hidden_layers = 0, vae_epochs = 0,
      max_epochs = 1, compute_variance = FALSE, show_progress = FALSE
    ),
    "epsilon must be finite and nonnegative"
  )
  omitted <- m_lbcnet(
    Z = d$Z, Tr = Tr_missing, na.action = na.omit,
    ck = 0.5, h = 0.3, hidden_dim = 2, num_hidden_layers = 0,
    vae_epochs = 0, max_epochs = 1, compute_variance = FALSE,
    show_progress = FALSE
  )
  expect_equal(nrow(omitted$Z), nrow(d$Z) - 1L)
})


test_that("joint outcome inference retains covariance and pairwise identities", {
  fit <- get_m_lbcnet_test_fit("outcome")
  n <- nrow(fit$Z)

  expect_equal(nrow(fit$means), 3)
  expect_equal(dim(fit$covariance), c(3, 3))
  expect_equal(fit$covariance, t(fit$covariance), tolerance = 1e-8)
  expect_true(all(is.finite(fit$means$se)))
  expect_true(all(fit$means$se >= 0))
  expect_equal(nrow(fit$pairwise_ate), choose(3, 2))

  for (row in seq_len(nrow(fit$pairwise_ate))) {
    item <- fit$pairwise_ate[row, ]
    index_1 <- match(item$treatment_1, fit$means$treatment)
    index_2 <- match(item$treatment_2, fit$means$treatment)
    expect_equal(
      item$estimate,
      fit$means$estimate[index_1] - fit$means$estimate[index_2],
      tolerance = 1e-6
    )
    expected_variance <- fit$covariance[index_1, index_1] +
      fit$covariance[index_2, index_2] -
      2 * fit$covariance[index_1, index_2]
    expect_equal(item$se^2, expected_variance, tolerance = 1e-6)
  }

  expect_equal(dim(fit$influence_functions), c(n, 3))
  expect_equal(unlist(fit$inference_diagnostics$H_shape), c(3, 21))
  expect_true(is.finite(
    fit$inference_diagnostics$minimum_H_row_distance
  ))
  expect_gt(fit$inference_diagnostics$minimum_H_row_distance, 1e-8)
  centered_if <- sweep(
    fit$influence_functions, 2, colMeans(fit$influence_functions)
  )
  reconstructed <- crossprod(centered_if) / n^2
  expect_equal(reconstructed, fit$covariance, tolerance = 1e-6)
})


test_that("epsilon flooring preserves the simplex and epsilon zero works", {
  fit <- get_m_lbcnet_test_fit("gps")
  epsilon <- fit$parameters$epsilon
  expect_true(all(fit$fitted.values >= epsilon - 1e-7))
  expect_equal(rowSums(fit$fitted.values), rep(1, nrow(fit$Z)),
               tolerance = 1e-6)

  skip_if_no_m_lbcnet_python()
  d <- m_lbcnet_test_data
  zero_fit <- m_lbcnet(
    Z = d$Z, Tr = d$Tr, ck = 0.5, h = 0.3,
    hidden_dim = 2, num_hidden_layers = 0,
    vae_epochs = 0, max_epochs = 1, epsilon = 0,
    compute_variance = FALSE, show_progress = FALSE, seed = 17
  )
  expect_true(all(is.finite(zero_fit$fitted.values)))
  expect_true(all(zero_fit$fitted.values > 0))
  expect_equal(rowSums(zero_fit$fitted.values), rep(1, nrow(d$Z)),
               tolerance = 1e-6)
})


test_that("M-LBCNet extraction, print, and summary methods are separate", {
  fit_gps <- get_m_lbcnet_test_fit("gps")
  fit_y <- get_m_lbcnet_test_fit("outcome")

  expect_equal(getLBC(fit_gps, "fitted.values"), fit_gps$fitted.values)
  expect_identical(
    getLBC(fit_gps, "bandwidth_pilot_method"), "not_used"
  )
  extracted <- getLBC(
    fit_y, c("means", "covariance", "pairwise_ate")
  )
  expect_named(extracted, c("means", "covariance", "pairwise_ate"))
  expect_identical(getLBC(fit_y, "ALL"), fit_y)
  expect_error(getLBC(fit_y, "not_a_component"), "Invalid component")

  expect_output(print(fit_gps), "M-LBCNet Model")
  expect_output(print(fit_gps), "Bandwidth Pilot: not_used")
  expect_output(print(fit_gps), "only the joint GPS", ignore.case = TRUE)
  expect_output(print(fit_y), "Treatment-Specific Marginal Means")
  expect_output(print(fit_y), "Pairwise ATEs")

  summary_result <- NULL
  expect_output(
    summary_result <- summary(fit_y),
    "GPS Summary by Treatment"
  )
  expect_true(is.list(summary_result))
  expect_equal(summary_result$treatment_count, 3)
  expect_identical(summary_result$bandwidth_pilot_method, "not_used")
  expect_equal(dim(summary_result$gps_summary), c(3, 6))
})


test_that("binary GSD and LSD numerical behavior remains unchanged", {
  Z <- cbind(
    x = c(-2, -1, 0, 1, 2, 3, 4, 5),
    z = c(3, -1, 2, 0, 4, 1, 5, 2)
  )
  treatment <- c(0, 1, 0, 1, 0, 1, 0, 1)
  propensity <- c(0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
  weights <- 1 / (
    treatment * propensity + (1 - treatment) * (1 - propensity)
  )
  expected_gsd <- c(
    x = -48.9742054091139,
    z = -362.68166458875
  )
  expected_lsd <- matrix(
    c(
      36.6575799144803, 44.4922154888795,
      411.019918228972, 395.902814825282
    ),
    nrow = 2,
    dimnames = list(NULL, c("x", "z"))
  )
  binary_fit <- list(
    fitted.values = propensity,
    weights = weights,
    loss = 0.1234,
    lsd_train = list(
      lsd_max = 411.019918228972,
      lsd_mean = 222.018132114403
    ),
    parameters = list(
      hidden_dim = 3L, num_hidden_layers = 1L,
      vae_lr = 0.01, lr = 0.05,
      weight_decay = 1e-5, balance_lambda = 1
    ),
    estimand = "ATE",
    stopping_criteria = list(
      lsd_threshold = 2, rolling_window = 5L, max_epochs = 10L
    ),
    ate_flag = 1,
    seed = 1L,
    call = quote(lbc_net(Z = Z, Tr = treatment)),
    formula = treatment ~ x + z,
    Z = Z,
    Tr = treatment,
    ck = c(0.3, 0.7),
    h = c(0.2, 0.25),
    K = 2L,
    rho = 0.15,
    kernel = "gaussian",
    ps_logistic = propensity,
    effect = NULL,
    se = NULL,
    ci = NULL
  )
  class(binary_fit) <- "lbc_net"

  expect_equal(gsd(Z = Z, Tr = treatment, wt = weights), expected_gsd,
               tolerance = 1e-12)
  expect_equal(gsd(binary_fit), expected_gsd, tolerance = 1e-12)
  binary_lsd <- lsd(binary_fit)
  expect_s3_class(binary_lsd, "lsd")
  expect_named(
    binary_lsd,
    c(
      "LSD", "LSD_mean", "LSD_max", "ck", "h", "Z", "Tr", "K",
      "rho", "kernel", "ate_flag"
    )
  )
  expect_equal(binary_lsd$LSD, expected_lsd, tolerance = 1e-12)
  expect_equal(binary_lsd$LSD_mean, 222.018132114403, tolerance = 1e-12)
  expect_equal(binary_lsd$LSD_max, 411.019918228972, tolerance = 1e-12)

  expect_identical(getLBC(binary_fit, "fitted.values"), propensity)
  expect_output(print(binary_fit), "Max Training Epochs: 10", fixed = TRUE)
  binary_summary <- NULL
  expect_output(
    binary_summary <- summary(binary_fit),
    "Post-GSD",
    fixed = TRUE
  )
  expect_named(
    binary_summary,
    c(
      "sample_info", "loss", "local_balance", "balance_table",
      "treatment_effect", "gsd"
    )
  )
})


manual_pooled_ess_smd <- function(
    covariate, weights_1, weights_2, variance_floor = NULL) {
  mass_1 <- sum(weights_1)
  mass_2 <- sum(weights_2)
  mean_1 <- sum(weights_1 * covariate) / mass_1
  mean_2 <- sum(weights_2 * covariate) / mass_2
  variance_1 <- sum(weights_1 * (covariate - mean_1)^2) / mass_1
  variance_2 <- sum(weights_2 * (covariate - mean_2)^2) / mass_2
  if (!is.null(variance_floor)) {
    variance_1 <- max(variance_1, variance_floor)
    variance_2 <- max(variance_2, variance_floor)
  }
  ess_1 <- mass_1^2 / sum(weights_1^2)
  ess_2 <- mass_2^2 / sum(weights_2^2)
  100 * abs(mean_1 - mean_2) / sqrt(
    (ess_1 * variance_1 + ess_2 * variance_2) / (ess_1 + ess_2)
  )
}


diagnostic_key <- function(data, columns) {
  values <- lapply(data[columns], as.character)
  do.call(paste, c(values, sep = "\r"))
}


test_that("M-LBCNet GSD and LSD have the required keyed structure", {
  fit <- get_m_lbcnet_test_fit("gps")
  gsd_result <- gsd(fit)
  lsd_result <- lsd(fit)
  n_treatments <- fit$n_treatments
  n_covariates <- ncol(fit$Z)
  n_grid <- length(fit$ck)
  treatment_labels <- as.character(fit$treatment_levels)

  expect_named(gsd_result, c("versus_population", "pairwise"))
  expect_named(lsd_result, c("versus_population", "pairwise"))
  expect_named(
    gsd_result$versus_population,
    c("treatment", "covariate", "gsd")
  )
  expect_named(
    gsd_result$pairwise,
    c("treatment_1", "treatment_2", "covariate", "gsd")
  )
  expect_named(
    lsd_result$versus_population,
    c("treatment", "center", "covariate", "lsd")
  )
  expect_named(
    lsd_result$pairwise,
    c(
      "localizing_treatment", "treatment_1", "treatment_2",
      "center", "covariate", "lsd"
    )
  )

  expect_equal(
    nrow(gsd_result$versus_population),
    n_treatments * n_covariates
  )
  expect_equal(
    nrow(gsd_result$pairwise),
    choose(n_treatments, 2) * n_covariates
  )
  expect_equal(
    nrow(lsd_result$versus_population),
    n_treatments * n_grid * n_covariates
  )
  expect_equal(
    nrow(lsd_result$pairwise),
    n_treatments * choose(n_treatments, 2) * n_grid * n_covariates
  )

  treatment_pairs <- t(utils::combn(treatment_labels, 2L))
  expected_gsd_vp <- expand.grid(
    treatment = treatment_labels,
    covariate = colnames(fit$Z),
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
  )
  gsd_pair_index <- expand.grid(
    pair = seq_len(nrow(treatment_pairs)),
    covariate = colnames(fit$Z),
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
  )
  expected_gsd_pairwise <- data.frame(
    treatment_1 = treatment_pairs[gsd_pair_index$pair, 1L],
    treatment_2 = treatment_pairs[gsd_pair_index$pair, 2L],
    covariate = gsd_pair_index$covariate,
    stringsAsFactors = FALSE
  )
  expected_lsd_vp <- expand.grid(
    treatment = treatment_labels,
    center = fit$ck,
    covariate = colnames(fit$Z),
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
  )
  lsd_pair_index <- expand.grid(
    localizing_treatment = treatment_labels,
    pair = seq_len(nrow(treatment_pairs)),
    center = fit$ck,
    covariate = colnames(fit$Z),
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
  )
  expected_lsd_pairwise <- data.frame(
    localizing_treatment = lsd_pair_index$localizing_treatment,
    treatment_1 = treatment_pairs[lsd_pair_index$pair, 1L],
    treatment_2 = treatment_pairs[lsd_pair_index$pair, 2L],
    center = lsd_pair_index$center,
    covariate = lsd_pair_index$covariate,
    stringsAsFactors = FALSE
  )
  keyed_results <- list(
    list(gsd_result$versus_population, expected_gsd_vp,
         c("treatment", "covariate")),
    list(gsd_result$pairwise, expected_gsd_pairwise,
         c("treatment_1", "treatment_2", "covariate")),
    list(lsd_result$versus_population, expected_lsd_vp,
         c("treatment", "center", "covariate")),
    list(lsd_result$pairwise, expected_lsd_pairwise,
         c(
           "localizing_treatment", "treatment_1", "treatment_2",
           "center", "covariate"
         ))
  )
  for (keyed_result in keyed_results) {
    observed_keys <- diagnostic_key(keyed_result[[1L]], keyed_result[[3L]])
    expected_keys <- diagnostic_key(keyed_result[[2L]], keyed_result[[3L]])
    expect_identical(anyDuplicated(observed_keys), 0L)
    expect_setequal(observed_keys, expected_keys)
  }

  expect_setequal(
    as.character(unique(gsd_result$versus_population$treatment)),
    treatment_labels
  )
  expect_setequal(
    as.character(unique(gsd_result$pairwise$treatment_1)),
    treatment_labels[-n_treatments]
  )
  expect_setequal(
    unique(c(
      as.character(gsd_result$pairwise$treatment_1),
      as.character(gsd_result$pairwise$treatment_2)
    )),
    treatment_labels
  )
  expect_setequal(
    as.character(unique(lsd_result$versus_population$treatment)),
    treatment_labels
  )
  expect_setequal(
    as.character(unique(lsd_result$pairwise$localizing_treatment)),
    treatment_labels
  )
  expect_setequal(
    unique(c(
      as.character(lsd_result$pairwise$treatment_1),
      as.character(lsd_result$pairwise$treatment_2)
    )),
    treatment_labels
  )
  expect_setequal(
    lsd_result$versus_population$covariate,
    colnames(fit$Z)
  )
  expect_false("(Intercept)" %in% lsd_result$versus_population$covariate)
  expect_identical(
    sort(unique(lsd_result$versus_population$center)),
    sort(as.numeric(fit$ck))
  )
  expect_identical(
    sort(unique(lsd_result$pairwise$center)),
    sort(as.numeric(fit$ck))
  )
  expect_true(all(is.finite(gsd_result$versus_population$gsd)))
  expect_true(all(is.finite(gsd_result$pairwise$gsd)))
  expect_true(all(is.finite(lsd_result$versus_population$lsd)))
  expect_true(all(is.finite(lsd_result$pairwise$lsd)))
  expect_true(all(gsd_result$versus_population$gsd >= 0))
  expect_true(all(gsd_result$pairwise$gsd >= 0))
  expect_true(all(lsd_result$versus_population$lsd >= 0))
  expect_true(all(lsd_result$pairwise$lsd >= 0))
})


test_that("M-LBCNet diagnostics use the pooled variance and ESS formula", {
  fit <- get_m_lbcnet_test_fit("gps")
  global <- gsd(fit)
  local <- lsd(fit)$versus_population

  treatment_index <- 2L
  covariate_index <- 1L
  treatment_label <- as.character(fit$treatment_levels[treatment_index])
  arm_weights <- (fit$Tr_code == treatment_index - 1L) /
    fit$fitted.values[, treatment_index]
  reported_gsd_vp <- global$versus_population$gsd[
    as.character(global$versus_population$treatment) == treatment_label &
      global$versus_population$covariate == colnames(fit$Z)[covariate_index]
  ]
  expected_gsd_vp <- manual_pooled_ess_smd(
    fit$Z[, covariate_index], arm_weights, rep(1, nrow(fit$Z))
  )
  expect_equal(reported_gsd_vp, expected_gsd_vp, tolerance = 1e-12)

  treatment_1 <- 1L
  treatment_2 <- 3L
  covariate_index <- 2L
  pair_weights_1 <- (fit$Tr_code == treatment_1 - 1L) /
    fit$fitted.values[, treatment_1]
  pair_weights_2 <- (fit$Tr_code == treatment_2 - 1L) /
    fit$fitted.values[, treatment_2]
  reported_gsd_pair <- global$pairwise$gsd[
    as.character(global$pairwise$treatment_1) ==
      as.character(fit$treatment_levels[treatment_1]) &
      as.character(global$pairwise$treatment_2) ==
      as.character(fit$treatment_levels[treatment_2]) &
      global$pairwise$covariate == colnames(fit$Z)[covariate_index]
  ]
  expected_gsd_pair <- manual_pooled_ess_smd(
    fit$Z[, covariate_index], pair_weights_1, pair_weights_2
  )
  expect_equal(reported_gsd_pair, expected_gsd_pair, tolerance = 1e-12)

  treatment_index <- 3L
  center_index <- 7L
  covariate_index <- 1L
  neighborhood <- stats::dnorm(
    (fit$fitted.values[, treatment_index] - fit$ck[center_index]) /
      fit$h[treatment_index, center_index]
  ) / fit$h[treatment_index, center_index]
  local_arm_weights <- neighborhood *
    (fit$Tr_code == treatment_index - 1L) /
    fit$fitted.values[, treatment_index]
  reported_lsd_vp <- local$lsd[
    as.character(local$treatment) ==
      as.character(fit$treatment_levels[treatment_index]) &
      local$center == fit$ck[center_index] &
      local$covariate == colnames(fit$Z)[covariate_index]
  ]
  expected_lsd_vp <- manual_pooled_ess_smd(
    fit$Z[, covariate_index], local_arm_weights, neighborhood,
    variance_floor = 1e-8
  )
  expect_equal(reported_lsd_vp, expected_lsd_vp, tolerance = 1e-10)
})


test_that("M-LBCNet pairwise GSD is invariant to treatment ordering", {
  fit <- get_m_lbcnet_test_fit("gps")
  original <- gsd(fit)$pairwise
  permutation <- c(2L, 1L, 3L)
  swapped <- fit
  swapped$fitted.values <- fit$fitted.values[, permutation, drop = FALSE]
  swapped$treatment_levels <- fit$treatment_levels[permutation]
  swapped$Tr_code <- match(fit$Tr_code, permutation - 1L) - 1L
  swapped$Tr <- factor(
    swapped$treatment_levels[swapped$Tr_code + 1L],
    levels = swapped$treatment_levels
  )
  reordered <- gsd(swapped)$pairwise

  label_1 <- as.character(fit$treatment_levels[1L])
  label_2 <- as.character(fit$treatment_levels[2L])
  original_pair <- original[
    original$treatment_1 == label_1 & original$treatment_2 == label_2,
    , drop = FALSE
  ]
  reordered_pair <- reordered[
    reordered$treatment_1 == label_2 & reordered$treatment_2 == label_1,
    , drop = FALSE
  ]
  original_pair <- original_pair[order(original_pair$covariate), ]
  reordered_pair <- reordered_pair[order(reordered_pair$covariate), ]
  expect_equal(original_pair$gsd, reordered_pair$gsd, tolerance = 1e-12)

  weights_1 <- (fit$Tr_code == 0L) / fit$fitted.values[, 1L]
  weights_2 <- (fit$Tr_code == 1L) / fit$fitted.values[, 2L]
  raw_difference <-
    sum(weights_1 * fit$Z[, 1L]) / sum(weights_1) -
    sum(weights_2 * fit$Z[, 1L]) / sum(weights_2)
  reordered_difference <-
    sum(weights_2 * fit$Z[, 1L]) / sum(weights_2) -
    sum(weights_1 * fit$Z[, 1L]) / sum(weights_1)
  expect_equal(raw_difference, -reordered_difference, tolerance = 1e-14)
})


test_that("public R and training Python M-LBCNet LSD agree by key", {
  fit <- get_m_lbcnet_test_fit("gps")
  public <- lsd(fit)$versus_population
  index <- expand.grid(
    treatment_index = seq_len(fit$n_treatments),
    center_index = seq_len(length(fit$ck)),
    covariate_index = seq_len(ncol(fit$Z)),
    KEEP.OUT.ATTRS = FALSE
  )
  python_values <- data.frame(
    treatment = fit$treatment_levels[index$treatment_index],
    center = fit$ck[index$center_index],
    covariate = colnames(fit$Z)[index$covariate_index],
    python_lsd = as.numeric(fit$lsd_values[cbind(
      index$treatment_index,
      index$center_index,
      index$covariate_index
    )]),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  public_key <- do.call(paste, c(public[c(
    "treatment", "center", "covariate"
  )], sep = "\r"))
  python_key <- do.call(paste, c(python_values[c(
    "treatment", "center", "covariate"
  )], sep = "\r"))
  expect_identical(anyDuplicated(public_key), 0L)
  expect_identical(anyDuplicated(python_key), 0L)

  comparison <- merge(
    public, python_values,
    by = c("treatment", "center", "covariate"),
    all = TRUE, sort = FALSE
  )
  expect_equal(
    nrow(comparison),
    fit$n_treatments * length(fit$ck) * ncol(fit$Z)
  )
  expect_false(anyNA(comparison$lsd))
  expect_false(anyNA(comparison$python_lsd))
  expect_lt(max(abs(comparison$lsd - comparison$python_lsd)), 1e-4)
  expect_lt(abs(max(public$lsd) - fit$lsd_train$lsd_max), 1e-4)
  expect_lt(abs(mean(public$lsd) - fit$lsd_train$lsd_mean), 1e-4)

  for (treatment_index in seq_len(fit$n_treatments)) {
    treatment_values <- public$lsd[
      public$treatment == fit$treatment_levels[treatment_index]
    ]
    expect_lt(abs(
      max(treatment_values) -
      fit$lsd_train$lsd_by_treatment$max_lsd[treatment_index]
    ), 1e-4)
    expect_lt(abs(
      mean(treatment_values) -
      fit$lsd_train$lsd_by_treatment$mean_lsd[treatment_index]
    ), 1e-4)
  }
})


test_that("pairwise LSD uses one common localizing GPS component", {
  treatment_levels <- c("usual care", "low dose", "high dose")
  treatment_code <- rep(0:2, each = 4)
  Z <- matrix(
    c(-3, -1, 2, 5, 4, 1, -2, -4, 0.5, 1.5, 2.5, 3.5),
    ncol = 1,
    dimnames = list(NULL, "x")
  )
  gps <- rbind(
    c(0.15, 0.65, 0.20), c(0.25, 0.53, 0.22),
    c(0.35, 0.41, 0.24), c(0.45, 0.29, 0.26),
    c(0.60, 0.15, 0.25), c(0.50, 0.23, 0.27),
    c(0.40, 0.31, 0.29), c(0.30, 0.39, 0.31),
    c(0.10, 0.20, 0.70), c(0.20, 0.20, 0.60),
    c(0.30, 0.20, 0.50), c(0.40, 0.20, 0.40)
  )
  colnames(gps) <- treatment_levels
  mock <- list(
    Z = Z,
    Tr = factor(treatment_levels[treatment_code + 1L],
                levels = treatment_levels),
    Tr_code = treatment_code,
    fitted.values = gps,
    treatment_levels = treatment_levels,
    n_treatments = 3L,
    ck = 0.5,
    h = matrix(c(0.08, 0.15, 0.28), nrow = 3L, ncol = 1L),
    K = 1L,
    kernel = "gaussian",
    weights = 1 / gps[cbind(seq_len(nrow(gps)), treatment_code + 1L)]
  )
  class(mock) <- "m_lbcnet"

  pooled_smd <- function(weights_1, weights_2) {
    x <- Z[, 1L]
    mean_1 <- sum(weights_1 * x) / sum(weights_1)
    mean_2 <- sum(weights_2 * x) / sum(weights_2)
    variance_1 <- max(
      sum(weights_1 * (x - mean_1)^2) / sum(weights_1), 1e-8
    )
    variance_2 <- max(
      sum(weights_2 * (x - mean_2)^2) / sum(weights_2), 1e-8
    )
    ess_1 <- sum(weights_1)^2 / sum(weights_1^2)
    ess_2 <- sum(weights_2)^2 / sum(weights_2^2)
    100 * abs(mean_1 - mean_2) / sqrt(
      (ess_1 * variance_1 + ess_2 * variance_2) / (ess_1 + ess_2)
    )
  }
  gaussian_weights <- function(probability, bandwidth) {
    exp(-((probability - 0.5) / bandwidth)^2 / 2) /
      (sqrt(2 * pi) * bandwidth)
  }

  pairwise <- lsd(mock)$pairwise
  reported <- pairwise$lsd[
    pairwise$localizing_treatment == "high dose" &
    pairwise$treatment_1 == "usual care" &
    pairwise$treatment_2 == "low dose" &
    pairwise$covariate == "x"
  ]
  common_neighborhood <- gaussian_weights(gps[, 3L], mock$h[3L, 1L])
  expected <- pooled_smd(
    common_neighborhood * (treatment_code == 0L) / gps[, 1L],
    common_neighborhood * (treatment_code == 1L) / gps[, 2L]
  )
  wrong_separate_localizer <- pooled_smd(
    common_neighborhood * (treatment_code == 0L) / gps[, 1L],
    gaussian_weights(gps[, 2L], mock$h[2L, 1L]) *
      (treatment_code == 1L) / gps[, 2L]
  )
  wrong_localizer_bandwidth <- pooled_smd(
    gaussian_weights(gps[, 3L], mock$h[2L, 1L]) *
      (treatment_code == 0L) / gps[, 1L],
    gaussian_weights(gps[, 3L], mock$h[2L, 1L]) *
      (treatment_code == 1L) / gps[, 2L]
  )
  expect_equal(reported, expected, tolerance = 1e-12)
  expect_gt(abs(reported - wrong_separate_localizer), 1)
  expect_gt(abs(reported - wrong_localizer_bandwidth), 1e-6)

  unsupported <- mock
  unsupported$ck <- 0.99
  unsupported$h <- matrix(0.001, nrow = 3L, ncol = 1L)
  unsupported$kernel <- "uniform"
  unsupported_lsd <- lsd(unsupported)
  expect_true(all(
    unsupported_lsd$versus_population$lsd == 1e8
  ))
  expect_true(all(unsupported_lsd$pairwise$lsd == 1e8))
})


test_that("R and Python use the same unsupported-neighborhood sentinel", {
  skip_if_no_m_lbcnet_python()
  gps <- rbind(
    c(0.60, 0.25, 0.15), c(0.50, 0.30, 0.20),
    c(0.25, 0.55, 0.20), c(0.20, 0.50, 0.30),
    c(0.15, 0.25, 0.60), c(0.20, 0.30, 0.50)
  )
  treatment_code <- rep(0:2, each = 2L)
  Z <- matrix(seq_len(nrow(gps)), ncol = 1L,
              dimnames = list(NULL, "x"))
  h <- matrix(0.001, nrow = 3L, ncol = 1L)
  mock <- list(
    Z = Z,
    Tr_code = treatment_code,
    fitted.values = gps,
    treatment_levels = c("A", "B", "C"),
    n_treatments = 3L,
    ck = 0.99,
    h = h,
    K = 1L,
    kernel = "uniform"
  )
  class(mock) <- "m_lbcnet"
  r_values <- lsd(mock)$versus_population$lsd

  module <- reticulate::import_from_path(
    "m_lbcnet", path = system.file("python", package = "LBCNet"),
    convert = FALSE
  )
  torch <- reticulate::import("torch", convert = FALSE)
  python_result <- module$m_lbcnet_lsd(
    torch$tensor(gps, dtype = torch$float32),
    torch$tensor(treatment_code, dtype = torch$long),
    torch$tensor(Z, dtype = torch$float32),
    torch$tensor(0.99, dtype = torch$float32),
    torch$tensor(h, dtype = torch$float32),
    kernel_id = 1L
  )
  python_values <- reticulate::py_to_r(
    reticulate::py_get_item(python_result, 3L)$detach()$cpu()$numpy()
  )

  expect_true(all(r_values == 1e8))
  expect_true(all(python_values == 1e8))
  expect_equal(as.numeric(python_values), r_values, tolerance = 0)
})


test_that("relocated M-LBCNet methods dispatch and summarize diagnostics", {
  fit <- get_m_lbcnet_test_fit("gps")

  print_visible <- NULL
  print_output <- capture.output(
    print_visible <- withVisible(print(fit))
  )
  expect_false(print_visible$visible)
  expect_identical(print_visible$value, fit)
  expect_true(any(grepl("Epochs Run:", print_output, fixed = TRUE)))
  expect_true(any(grepl(
    "LSD Criterion Achieved: No", print_output, fixed = TRUE
  )))
  required_print_fields <- c(
    "M-LBCNet Model", "Call:", "Sample Size:", "Treatment Groups:",
    "Group Sizes:", "Final Loss:", "Treatment-vs-Population Max LSD:",
    "Treatment-vs-Population Mean LSD:", "K:", "Kernel:",
    "Bandwidth Pilot:", "Hidden Layers:", "Hidden Units:",
    "Learning rates:", "Weight Decay:", "Balance Lambda:", "Epsilon:",
    "Epochs Run:", "LSD Threshold:", "Rolling Window:"
  )
  for (field in required_print_fields) {
    expect_true(any(grepl(field, print_output, fixed = TRUE)), info = field)
  }
  expect_false(any(grepl("converg", print_output, ignore.case = TRUE)))
  expect_false(any(grepl(
    "\\bcovariate\\b|localizing_treatment",
    print_output, ignore.case = TRUE, perl = TRUE
  )))

  summary_visible <- NULL
  summary_output <- capture.output(
    summary_visible <- withVisible(summary(fit))
  )
  expect_false(summary_visible$visible)
  summary_result <- summary_visible$value
  expect_named(summary_result, c(
    "call", "sample_size", "covariate_count", "treatment_count",
    "treatment_group_sizes", "bandwidth_pilot_method", "loss",
    "max_lsd", "mean_lsd", "lsd_by_treatment", "gps_summary",
    "observed_ipw_summary", "means", "pairwise_ate", "covariance",
    "global_balance", "local_balance", "gsd", "lsd"
  ))
  expect_named(summary_result$gsd, c("versus_population", "pairwise"))
  expect_named(summary_result$lsd, c("versus_population", "pairwise"))
  expected_gsd <- gsd(fit)
  expected_lsd <- lsd(fit)
  expect_equal(summary_result$gsd, expected_gsd, tolerance = 0)
  expect_equal(summary_result$lsd, expected_lsd, tolerance = 0)
  expect_equal(
    summary_result$global_balance$versus_population$max,
    max(expected_gsd$versus_population$gsd), tolerance = 0
  )
  expect_equal(
    summary_result$global_balance$versus_population$mean,
    mean(expected_gsd$versus_population$gsd), tolerance = 0
  )
  expect_equal(
    summary_result$global_balance$pairwise$max,
    max(expected_gsd$pairwise$gsd), tolerance = 0
  )
  expect_equal(
    summary_result$local_balance$versus_population$max,
    max(expected_lsd$versus_population$lsd), tolerance = 0
  )
  expect_equal(
    summary_result$local_balance$pairwise$mean,
    mean(expected_lsd$pairwise$lsd), tolerance = 0
  )
  expect_true(any(grepl("Global Balance:", summary_output, fixed = TRUE)))
  expect_true(any(grepl("Local Balance:", summary_output, fixed = TRUE)))
  expect_false(any(grepl(
    "localizing_treatment", summary_output, fixed = TRUE
  )))
  expect_false(any(grepl(
    "\\bcovariate\\b", summary_output, ignore.case = TRUE, perl = TRUE
  )))

  fit_y <- get_m_lbcnet_test_fit("outcome")
  outcome_summary <- NULL
  invisible(capture.output(outcome_summary <- summary(fit_y)))
  expect_identical(outcome_summary$means, fit_y$means)
  expect_identical(outcome_summary$pairwise_ate, fit_y$pairwise_ate)
  expect_identical(outcome_summary$covariance, fit_y$covariance)

  expect_identical(getLBC(fit, "lsd_values"), fit$lsd_values)
  expect_false("gsd" %in% names(getLBC(fit, "ALL")))
  expect_false("lsd" %in% names(getLBC(fit, "ALL")))
})
