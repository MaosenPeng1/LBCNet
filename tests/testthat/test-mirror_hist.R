test_that("mirror_hist correctly generates a ggplot2 object", {
  set.seed(123)

  # Simulated Data
  ps <- runif(1000)  # Random propensity scores
  Tr <- sample(0:1, 1000, replace = TRUE)  # Random treatment assignment

  # ---- Valid Call ----
  plot <- mirror_hist(ps = ps, Tr = Tr, bins = 50, size = 0.8)
  expect_s3_class(plot, "ggplot")  # Must be a ggplot object

  # ---- Valid Customization Tests ----
  expect_s3_class(mirror_hist(ps = ps, Tr = Tr, bins = 100), "ggplot")
  expect_s3_class(mirror_hist(ps = ps, Tr = Tr, size = 1.2), "ggplot")
  expect_s3_class(mirror_hist(ps = ps, Tr = Tr, theme.size = 10), "ggplot")

  # ---- Error Handling ----
  expect_error(mirror_hist(Tr = Tr), "Must provide", fixed = TRUE)
  expect_error(mirror_hist(ps = ps), "Must provide", fixed = TRUE)
  expect_error(mirror_hist(object = list()), "must be of class 'lbc_net'", fixed = TRUE)  # Wrong object class

  # ---- Testing with `lbc_net` Object ----
  set.seed(123)
  n <- 500
  Z <- MASS::mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
  Tr <- rbinom(n, 1, 0.5)

  # Run with custom parameters
  model <- lbc_net(Z = Z, Tr = Tr, hidden_dim = 50, max_epochs = 1000)
  plot_model <- mirror_hist(model)
  expect_s3_class(plot_model, "ggplot")
})
