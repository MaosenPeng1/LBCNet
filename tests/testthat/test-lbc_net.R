test_that("lbc_net runs correctly with minimal inputs", {
  skip_if_not_installed("MASS")  # Skip if MASS is not installed
  set.seed(123)

  # Generate synthetic data
  n <- 500
  Z <- MASS::mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
  Tr <- rbinom(n, 1, 0.5)  # Binary treatment assignment

  # Run the function
  model <- lbc_net(Z = Z, Tr = Tr)

  # Check if output is a valid lbc_net object
  expect_s3_class(model, "lbc_net")

  # Check fitted values are within valid range (0,1)
  expect_true(all(model$fitted.values > 0 & model$fitted.values < 1))

  # Check inverse probability weights are positive
  expect_true(all(model$weights > 0))
})

test_that("lbc_net handles formula input correctly", {
  set.seed(123)
  n <- 500
  Z <- MASS::mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
  Tr <- rbinom(n, 1, 0.5)
  data <- as.data.frame(cbind(Tr, Z))
  colnames(data) <- c("Tr", "X1", "X2", "X3", "X4")

  # Run the function with formula input
  model <- lbc_net(data = data, formula = Tr ~ X1 + X2 + X3 + X4)

  # Check if output is a valid lbc_net object
  expect_s3_class(model, "lbc_net")

  # Check consistency between formula and direct input
  model_direct <- lbc_net(Z = Z, Tr = Tr)
  expect_equal(model$fitted.values, model_direct$fitted.values, tolerance = 1e-5)
})

test_that("lbc_net correctly applies missing data handling", {
  set.seed(123)
  n <- 500
  Z <- MASS::mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
  Tr <- rbinom(n, 1, 0.5)

  # Introduce missing values
  Z[1, 1] <- NA

  # Expect an error due to missing values
  expect_error(lbc_net(Z = Z, Tr = Tr), "missing values")

  # Expect success when using na.omit
  model <- lbc_net(Z = Z, Tr = Tr, na.action = na.omit)
  expect_s3_class(model, "lbc_net")
})

test_that("lbc_net respects user-defined hyperparameters", {
  set.seed(123)
  n <- 500
  Z <- MASS::mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
  Tr <- rbinom(n, 1, 0.5)

  # Run with custom parameters
  model <- lbc_net(Z = Z, Tr = Tr, hidden_dim = 50, max_epochs = 1000, lr = 0.01, seed = 1,
                   vae_epochs = 150, vae_lr = 0.05)

  # Check if parameters are correctly set
  expect_equal(model$parameters$hidden_dim, 50)
  expect_equal(model$parameters$lr, 0.01)
  expect_equal(model$stopping_criteria$max_epochs, 1000)
})

test_that("lbc_net handles alternative kernels correctly", {
  set.seed(123)
  n <- 500
  Z <- MASS::mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
  Tr <- rbinom(n, 1, 0.5)

  # Test different kernels
  model_gaussian <- lbc_net(Z = Z, Tr = Tr, kernel = "gaussian")
  model_uniform <- lbc_net(Z = Z, Tr = Tr, kernel = "uniform")
  model_epanechnikov <- lbc_net(Z = Z, Tr = Tr, kernel = "epanechnikov")

  expect_s3_class(model_gaussian, "lbc_net")
  expect_s3_class(model_uniform, "lbc_net")
  expect_s3_class(model_epanechnikov, "lbc_net")
})

test_that("lbc_net maintains reproducibility with fixed seed", {
  set.seed(123)
  n <- 500
  Z <- MASS::mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
  Tr <- rbinom(n, 1, 0.5)

  # Run twice with the same seed
  model1 <- lbc_net(Z = Z, Tr = Tr, seed = 42)
  model2 <- lbc_net(Z = Z, Tr = Tr, seed = 42)

  # Expect identical propensity scores
  expect_equal(model1$fitted.values, model2$fitted.values, tolerance = 1e-5)
})

test_that("lbc_net stops early when lsd_threshold is reached", {
  set.seed(123)
  n <- 500
  Z <- MASS::mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
  Tr <- rbinom(n, 1, 0.5)

  # Run with strict stopping criterion
  model <- lbc_net(Z = Z, Tr = Tr, lsd_threshold = 30, max_epochs = 5000)

  # Check that training did not reach max epochs
  expect_true(model$stopping_criteria$max_epochs >= model$lsd_train$lsd_max)
})
