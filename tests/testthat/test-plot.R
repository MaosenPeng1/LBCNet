test_that("plot.lsd correctly generates a ggplot2 object", {
  set.seed(123)
  n <- 500
  Z <- MASS::mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
  Tr <- rbinom(n, 1, 0.5)

  # Run with custom parameters
  model <- lbc_net(Z = Z, Tr = Tr, hidden_dim = 50, max_epochs = 1000)
  lsd_obj <- lsd(model)
  class(lsd_obj) <- "lsd"

  # ---- Valid Calls ----
  p1 <- plot(lsd_obj)
  expect_s3_class(p1, "ggplot")  # Default should work

  p2 <- plot(lsd_obj, cov = "ALL")
  expect_s3_class(p2, "ggplot")

  p3 <- plot(lsd_obj, cov = 1)
  expect_s3_class(p3, "ggplot")

  p4 <- plot(lsd_obj, cov = 2)
  expect_s3_class(p4, "ggplot")

  # ---- Customization ----
  expect_s3_class(plot(lsd_obj, point.color = "red", point.size = 2, line.size = 1), "ggplot")
  expect_s3_class(plot(lsd_obj, cov = "ALL", box.loc = seq(0.2, 0.8, by = 0.1)), "ggplot")

  # ---- Error Handling ----
  expect_error(plot.lsd(list()), "must be of class 'lsd'", fixed = TRUE)
  expect_error(plot(lsd_obj, cov = "CovNotFound"), "not found", fixed = TRUE)  # Invalid name
  expect_error(plot(lsd_obj, cov = 99), "out of range", fixed = TRUE)  # Out-of-range index
})
