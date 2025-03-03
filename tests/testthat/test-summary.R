test_that("summary.lbc_net prints expected output", {
  set.seed(123)

  # Create a mock lbc_net object
  mock_lbc_net <- list(
    formula = as.formula("Tr ~ X1 + X2"),
    Z = matrix(rnorm(100), nrow = 50, ncol = 2),
    Tr = sample(0:1, 50, replace = TRUE),
    losses = list(balance_loss = 0.05, calibration_loss = 0.02, total_loss = 0.07),
    lsd_train = list(lsd_max = 1.2, lsd_mean = 0.6),
    ATE = 1,
    weights = runif(100, 0.1, 0.9)
  )
  class(mock_lbc_net) <- "lbc_net"

  # Capture printed output
  expect_output(summary(mock_lbc_net), "Sample Size: 50")
  expect_output(summary(mock_lbc_net), "Number of Covariates: 2")
  expect_output(summary(mock_lbc_net), "Balance Loss:      0.0500")
  expect_output(summary(mock_lbc_net), "Calibration Loss:  0.0200")
  expect_output(summary(mock_lbc_net), "Total Loss:        0.0700")
  expect_output(summary(mock_lbc_net), "Max LSD:   1.2000")
  expect_output(summary(mock_lbc_net), "Mean LSD:  0.6000")
})

test_that("summary.lbc_net handles missing Y correctly", {
  set.seed(123)

  mock_lbc_net <- list(
    formula = as.formula("Tr ~ X1 + X2"),
    Z = matrix(rnorm(100), nrow = 50, ncol = 2),
    Tr = sample(0:1, 50, replace = TRUE),
    losses = list(balance_loss = 0.05, calibration_loss = 0.02, total_loss = 0.07),
    lsd_train = list(lsd_max = 1.2, lsd_mean = 0.6),
    ATE = 1,
    weights = runif(100, 0.1, 0.9)
  )
  class(mock_lbc_net) <- "lbc_net"

  expect_output(summary(mock_lbc_net), "Effect estimate not calculated")
})

test_that("summary.lbc_net throws error for non-lbc_net objects", {
  expect_error(summary.lbc_net(list()), "Error: `object` must be an object of class 'lbc_net'.")
})

test_that("summary.lsd prints expected output", {
  set.seed(123)

  # Create a mock lsd object
  mock_lsd <- list(
    Z = matrix(rnorm(100), nrow = 50, ncol = 2),
    Tr = sample(0:1, 50, replace = TRUE),
    LSD_mean = 0.8,
    LSD_max = 1.5,
    LSD = matrix(runif(100, 0.5, 1.5), ncol = 2),
    ATE = 1,
    weights = runif(100, 0.1, 0.9)
  )
  class(mock_lsd) <- "lsd"

  expect_output(summary(mock_lsd), "Sample Size: 50")
  expect_output(summary(mock_lsd), "Number of Covariates: 2")
  expect_output(summary(mock_lsd), "Max LSD:   1.5000")
  expect_output(summary(mock_lsd), "Mean LSD:  0.8000")
})

test_that("summary.lsd throws error for non-lsd objects", {
  expect_error(summary.lsd(list()), "Error: `object` must be of class 'lsd'.")
})
