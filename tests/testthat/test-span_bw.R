library(testthat)

test_that("span_bw correctly computes adaptive bandwidths", {
  set.seed(123)

  # Generate a test dataset
  N <- 500
  Z <- as.data.frame(matrix(rnorm(N * 5), ncol = 5))
  colnames(Z) <- paste0("X", 1:5)
  Tr <- rbinom(N, 1, 0.5)  # Binary treatment assignment

  # Logistic regression to obtain propensity scores
  log.fit <- glm(Tr ~ ., data = Z, family = "binomial")
  p <- log.fit$fitted.values  # Extract estimated propensity scores

  # Set kernel center points and span
  ck <- seq(0.01, 0.99, 0.01)
  rho <- 0.15

  # Compute bandwidths
  h <- span_bw(rho, ck, p)

  # Check the length of output matches the length of ck
  expect_length(h, length(ck))

  # Check that computed bandwidths are numeric and positive
  expect_true(is.numeric(h))
  expect_true(all(h > 0))

})

test_that("span_bw handles edge cases correctly", {
  set.seed(123)
  p <- runif(100, 0.1, 0.9)  # Simulated valid propensity scores
  ck <- seq(0.05, 0.95, 0.1)
  rho <- 0.1

  h <- span_bw(rho, ck, p)

  expect_type(h, "double")  # Ensure output is a numeric vector
  expect_length(h, length(ck))
  expect_true(all(h > 0))  # Bandwidths should be strictly positive
})

test_that("span_bw throws errors for invalid inputs", {
  p <- runif(100, 0.1, 0.9)  # Simulated valid propensity scores
  ck <- seq(0.05, 0.95, 0.1)

  expect_error(span_bw(-0.1, ck, p), "Error: 'rho' must be a numeric value between 0 and 1.")
  expect_error(span_bw(1.2, ck, p), "Error: 'rho' must be a numeric value between 0 and 1.")
  expect_error(span_bw(0.1, c(-0.1, 0.2, 0.3), p), "Error: 'ck' must be a numeric vector with values in \\(0,1\\).")
  expect_error(span_bw(0.1, ck, c(0, 0.5, 1)), "Error: 'p' must be a numeric vector with values strictly between 0 and 1.")
})

