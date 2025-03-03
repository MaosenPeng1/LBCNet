test_that("gsd correctly calculates Global Standardized Mean Difference", {
  set.seed(123)

  # Generate example data
  Z <- matrix(rnorm(200), nrow = 100, ncol = 2)  # Covariates
  Tr <- rbinom(100, 1, 0.5)  # Treatment assignment
  ps <- runif(100, 0.1, 0.9)  # Simulated propensity scores

  # Convert ps to weights
  wt <- 1 / (Tr * ps + (1 - Tr) * (1 - ps))

  # --- TEST 1: GSD with explicitly provided wt ---
  gsd_values <- gsd(Z = Z, Tr = Tr, wt = wt)
  expect_type(gsd_values, "double")
  expect_length(gsd_values, ncol(Z))  # Should return 1 GSD per covariate
  expect_true(all(!is.na(gsd_values)))  # Ensure no NA values

  # --- TEST 2: GSD with automatically computed weights from ps ---
  gsd_values_ps <- gsd(Z = Z, Tr = Tr, ps = ps)
  expect_type(gsd_values_ps, "double")
  expect_equal(gsd_values, gsd_values_ps, tolerance = 1e-6)  # Should match explicitly computed weights

  # --- TEST 3: GSD with a single covariate vector ---
  gsd_single <- gsd(Z = Z[,1], Tr = Tr, wt = wt)
  expect_type(gsd_single, "double")
  expect_length(gsd_single, 1)

  # --- TEST 4: Error handling for invalid/missing inputs ---
  expect_error(gsd(Z = Z, Tr = Tr), "Must provide", fixed = TRUE)
  expect_error(gsd(Z = Z, wt = wt), "Must provide", fixed = TRUE)
  expect_error(gsd(Tr = Tr, wt = wt), "Must provide", fixed = TRUE)
  expect_error(gsd(Z = Z, Tr = Tr, ps = c(0.5, 1.5)), "strictly between 0 and 1.", fixed = TRUE)  # Invalid ps

  # --- TEST 5: GSD with an `lbc_net` object ---
  set.seed(123)
  n <- 500
  Z <- MASS::mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
  Tr <- rbinom(n, 1, 0.5)

  # Run with custom parameters
  model <- lbc_net(Z = Z, Tr = Tr, hidden_dim = 50, max_epochs = 1000)
  gsd_model <- gsd(model)
  expect_type(gsd_model, "double")
  expect_length(gsd_model, ncol(getLBC(model, "Z")))  # Should return GSD for each covariate
})
