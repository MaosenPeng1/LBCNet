test_that("getLBC correctly extracts components from lbc_net", {
  set.seed(123)
  n <- 500
  Z <- MASS::mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
  Tr <- rbinom(n, 1, 0.5)

  # Run with custom parameters
  model <- lbc_net(Z = Z, Tr = Tr, hidden_dim = 50, max_epochs = 1000)

  # Single valid component extraction
  fitted_values <- getLBC(model, "fitted.values")
  expect_type(fitted_values, "double")  # Should return numeric values

  # Multiple valid component extraction
  components <- getLBC(model, c("fitted.values", "weights"))
  expect_true(is.list(components))  # Should return a list
  expect_named(components, c("fitted.values", "weights"))

  # Extract all components
  all_components <- getLBC(model, "ALL")
  expect_true(is.list(all_components))  # Should return a full object as list
  expect_named(all_components)  # Check if components are named correctly

  # Invalid component extraction
  expect_error(getLBC(model, "invalid_component"), "Invalid component name")

  # Non-character name
  expect_error(getLBC(model, 1), "`name` must be a non-empty character vector.")

  # NULL object test
  expect_error(getLBC(NULL, "fitted.values"), "must be of class")
})

test_that("getLBC correctly extracts components from lsd objects", {
  set.seed(123)
  n <- 500
  Z <- MASS::mvrnorm(n, mu = rep(0, 4), Sigma = diag(4))
  Tr <- rbinom(n, 1, 0.5)

  # Run with custom parameters
  model <- lbc_net(Z = Z, Tr = Tr, hidden_dim = 50, max_epochs = 1000)
  lsd_model <- lsd(model)

  # Extracting a valid component
  lsd_values <- getLBC(lsd_model, "LSD")
  expect_true(is.matrix(lsd_values))  # LSD should be a matrix

  # Extract all components
  all_lsd_components <- getLBC(lsd_model, "ALL")
  expect_true(is.list(all_lsd_components))  # Should return a list

  # Extracting multiple valid components
  components <- getLBC(lsd_model, c("LSD", "LSD_mean"))
  expect_true(is.list(components))  # Should return a list
  expect_named(components, c("LSD", "LSD_mean"))

  # Invalid component extraction
  expect_error(getLBC(lsd_model, "invalid_component"), "Invalid component name")
})

