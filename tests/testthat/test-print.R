test_that("print.lbc_net correctly prints output", {
  # Mock lbc_net object
  mock_lbc_net <- list(
    formula = as.formula("Tr ~ X1 + X2"),
    Z = matrix(rnorm(100), nrow = 50, ncol = 2),
    Tr = sample(0:1, 50, replace = TRUE),
    loss = 0.0123,
    lsd_train = list(lsd_max = 3.45, lsd_mean = 1.23),
    parameters = list(hidden_dim = 50, num_hidden_layers = 2, vae_lr = 0.01, lr = 0.05, weight_decay = 1e-5, balance_lambda = 1.0),
    stopping_criteria = list(lsd_threshold = 2, rolling_window = 5, max_epochs = 1000),
    kernel = "gaussian",
    ate_flag = 1
  )
  class(mock_lbc_net) <- "lbc_net"

  expect_output(print(mock_lbc_net), "Sample Size: 50")
  expect_output(print(mock_lbc_net), "Treated:")
  expect_output(print(mock_lbc_net), "Final Loss Value: 0.0123")
  expect_output(print(mock_lbc_net), "Max LSD: 3.45%")
  expect_output(print(mock_lbc_net), "Mean LSD: 1.23%")
  expect_output(print(mock_lbc_net), "Hidden Layers: 2 | Hidden Units: 50")
  expect_output(print(mock_lbc_net), "LSD Threshold: 2.00% | Rolling Window: 5")
  expect_output(print(mock_lbc_net), "Max Training Epochs: 1000")
})

test_that("print.lbc_net throws an error for non-lbc_net objects", {
  expect_error(print.lbc_net(list()), "must be of class 'lbc_net'.", fixed = TRUE)
})

# ------------------------------------------------------------------------------------

test_that("print.lsd correctly prints output", {
  # Mock lsd object
  mock_lsd <- list(
    Z = matrix(rnorm(100), nrow = 50, ncol = 2),
    Tr = sample(0:1, 50, replace = TRUE),
    LSD_mean = 1.23,
    LSD_max = 3.45,
    kernel = "gaussian",
    ate_flag = 1
  )
  class(mock_lsd) <- "lsd"

  expect_output(print(mock_lsd), "Sample Size: 50")
  expect_output(print(mock_lsd), "Treated:")
  expect_output(print(mock_lsd), "Max LSD: 3.45")
  expect_output(print(mock_lsd), "Mean LSD: 1.23")
})

test_that("print.lsd throws an error for non-lsd objects", {
  expect_error(print.lsd(list()), "must be of class 'lsd'.", fixed = TRUE)
})
