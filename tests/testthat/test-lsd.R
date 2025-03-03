test_that("lsd correctly calculates Local Standardized Mean Difference", {
  set.seed(123)

  # Generate Example Data
  Z <- matrix(rnorm(200), nrow = 100, ncol = 2)  # Covariates
  Tr <- rbinom(100, 1, 0.5)  # Treatment assignment
  ps <- runif(100, 0.1, 0.9)  # Simulated propensity scores

  # Call `lsd()` with manually provided inputs
  lsd_result <- lsd(Z = Z, Tr = Tr, ps = ps, K = 99)

  # ---- Expected Structure Tests ----
  expect_s3_class(lsd_result, "lsd")  # Should return an `lsd` object
  expect_named(lsd_result, c("LSD", "LSD_mean", "LSD_max", "ck", "h", "Z", "Tr", "K", "rho", "kernel", "ATE"))

  # ---- Output Validity Tests ----
  expect_type(lsd_result$LSD, "double")  # LSD should be numeric
  expect_true(all(!is.na(lsd_result$LSD)))  # No missing values
  expect_true(lsd_result$LSD_max >= 0)  # Max LSD should be non-negative
  expect_true(lsd_result$LSD_mean >= 0)  # Mean LSD should be non-negative

  # ---- Kernel Tests ----
  lsd_gaussian <- lsd(Z = Z, Tr = Tr, ps = ps, kernel = "gaussian")
  lsd_uniform <- lsd(Z = Z, Tr = Tr, ps = ps, kernel = "uniform")
  lsd_epanechnikov <- lsd(Z = Z, Tr = Tr, ps = ps, kernel = "epanechnikov")

  expect_s3_class(lsd_gaussian, "lsd")
  expect_s3_class(lsd_uniform, "lsd")
  expect_s3_class(lsd_epanechnikov, "lsd")

  # ---- Error Handling Tests ----
  expect_error(lsd(Tr = Tr, ps = ps), "Must provide", fixed = TRUE)
  expect_error(lsd(Z = Z, ps = ps), "Must provide", fixed = TRUE)
  expect_error(lsd(Z = Z, Tr = Tr), "Must provide", fixed = TRUE)
  expect_error(lsd(Z = Z, Tr = Tr, ps = ps, kernel = "invalid_kernel"), "Invalid kernel specified", fixed = TRUE)

  # ---- Testing with `lbc_net` object ----
  set.seed(123)
  n <- 500
  Z <- MASS::mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
  Tr <- rbinom(n, 1, 0.5)

  # Run with custom parameters
  model <- lbc_net(Z = Z, Tr = Tr, hidden_dim = 50, max_epochs = 1000)
  lsd_model <- lsd(model)

  expect_s3_class(lsd_model, "lsd")
  expect_type(lsd_model$LSD, "double")
  expect_true(all(!is.na(lsd_model$LSD)))
})
