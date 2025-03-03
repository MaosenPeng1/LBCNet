# Load necessary functions
test_that("est_effect works correctly with manual inputs", {
  set.seed(123)

  # Simulated data
  Y <- rnorm(100)  # Outcomes
  Tr <- rbinom(100, 1, 0.5)  # Treatment assignment (binary)
  wt <- runif(100, 0.5, 1.5)  # Random inverse probability weights

  # Compute effects
  ate <- est_effect(Y = Y, Tr = Tr, wt = wt, type = "ATE")
  att <- est_effect(Y = Y, Tr = Tr, wt = wt, type = "ATT")
  mean_Y <- est_effect(Y = Y, Tr = Tr, wt = wt, type = "Y")

  # Check output types
  expect_type(ate, "double")
  expect_type(att, "double")
  expect_type(mean_Y, "double")

  # Check values are finite
  expect_true(is.finite(ate))
  expect_true(is.finite(att))
  expect_true(is.finite(mean_Y))
})

# Load precomputed lbc_net model
test_that("est_effect works correctly with an lbc_net object", {
  set.seed(123)
  n <- 500
  Z <- MASS::mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
  Tr <- rbinom(n, 1, 0.5)

  # Run with custom parameters
  model <- lbc_net(Z = Z, Tr = Tr, hidden_dim = 50, max_epochs = 1000)

  # Simulated data
  Y <- rnorm(500)

  # Compute effects
  ate <- est_effect(object = model, Y = Y, type = "ATE")
  att <- est_effect(object = model, Y = Y, type = "ATT")
  mean_Y <- est_effect(object = model, Y = Y, type = "Y")

  # Check output types
  expect_type(ate, "double")
  expect_type(att, "double")
  expect_type(mean_Y, "double")

  # Check values are finite
  expect_true(is.finite(ate))
  expect_true(is.finite(att))
  expect_true(is.finite(mean_Y))
})

# Test errors for incorrect inputs
test_that("est_effect handles errors correctly", {
  set.seed(123)

  Y <- rnorm(100)
  Tr <- rbinom(100, 1, 0.5)
  wt <- runif(100, 0.5, 1.5)

  expect_error(est_effect(Y = Y, Tr = Tr, type = "ATE"), "Must provide `Tr` and `wt` manually if `object` is NULL.")
  expect_error(est_effect(Y = Y, Tr = NULL, wt = wt, type = "ATE"), "Must provide `Tr` and `wt` manually if `object` is NULL.")
  expect_error(est_effect(Y = Y, Tr = Tr, wt = NULL, type = "ATE"), "Must provide `Tr` and `wt` manually if `object` is NULL.")
  expect_error(est_effect(Y = Y, Tr = Tr, wt = wt, type = "invalid"), "Error: `type` must be one of 'Y', 'ATE', or 'ATT'.")
})

# Edge case: all treated or all control
test_that("est_effect handles edge cases correctly", {
  set.seed(123)

  Y <- rnorm(100)
  Tr_all_1 <- rep(1, 100)  # All treated
  Tr_all_0 <- rep(0, 100)  # All control
  wt <- runif(100, 0.5, 1.5)

  # All treated: ATE should be NA
  expect_warning(ate_1 <- est_effect(Y = Y, Tr = Tr_all_1, wt = wt, type = "ATE"))
  expect_true(is.nan(ate_1) || is.na(ate_1))

  # All control: ATE should be NA
  expect_warning(ate_0 <- est_effect(Y = Y, Tr = Tr_all_0, wt = wt, type = "ATE"))
  expect_true(is.nan(ate_0) || is.na(ate_0))

  # Should still return a finite mean outcome for type = "Y"
  expect_true(is.finite(est_effect(Y = Y, Tr = Tr_all_1, wt = wt, type = "Y")))
  expect_true(is.finite(est_effect(Y = Y, Tr = Tr_all_0, wt = wt, type = "Y")))
})
