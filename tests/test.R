devtools::document()
devtools::load_all()

set.seed(123)

n <- 1000
X1 <- rnorm(n)
X2 <- rnorm(n)
X3 <- rnorm(n)

# Simple 3-treatment multinomial DGP
eta1 <-  0.4 * X1 - 0.2 * X2
eta2 <- -0.3 * X1 + 0.4 * X3
eta3 <- rep(0, n)

eta <- cbind(eta1, eta2, eta3)
exp_eta <- exp(eta - apply(eta, 1, max))
true_ps <- exp_eta / rowSums(exp_eta)

Tr <- apply(
  true_ps,
  1,
  function(p) sample(c("A", "B", "C"), size = 1, prob = p)
)

Tr <- factor(Tr, levels = c("A", "B", "C"))

dat <- data.frame(
  Tr = Tr,
  X1 = X1,
  X2 = X2,
  X3 = X3
)

fit <- m_lbcnet(
  data = dat,
  formula = Tr ~ X1 + X2 + X3,
  K = 19,                 # small first smoke test
  vae_epochs = 250,
  max_epochs = 5000,
  show_progress = TRUE
)

print(fit)
summary(fit)

dim(fit$fitted.values)
head(fit$fitted.values)

# Essential simplex checks
range(rowSums(fit$fitted.values))
max(abs(rowSums(fit$fitted.values) - 1))

# GPS should all be positive
range(fit$fitted.values)

# ATE weights
range(fit$weights)
all(is.finite(fit$weights))
all(fit$weights > 0)

# Treatment labels
colnames(fit$fitted.values)

# Training balance
fit$lsd_train

########################################################
set.seed(456)

tau <- c(A = 0, B = 1, C = 2)

dat$Y <-
  2 +
  0.5 * dat$X1 -
  0.3 * dat$X2 +
  tau[as.character(dat$Tr)] +
  rnorm(n)

fit_y <- m_lbcnet(
  data = dat,
  formula = Tr ~ X1 + X2 + X3,
  Y = dat$Y,
  K = 19,
  vae_epochs = 250,
  max_epochs = 5000,
  show_progress = FALSE
)

print(fit_y)
summary(fit_y)

fit_y$means
fit_y$covariance
fit_y$pairwise_ate

lsd(fit_y)
gsd(fit_y)

##############################################################3

set.seed(20)

# ------------------------------------------------------------------
# Simulation settings
# ------------------------------------------------------------------

B <- 50        # debugging first; later increase to 200 or 500
n <- 5000

true_mu <- c(
  A = 1,
  B = 2,
  C = 3
)

true_ate <- c(
  "A-B" = -1,
  "A-C" = -2,
  "B-C" = -1
)

# Storage
mean_results <- vector("list", B)
ate_results  <- vector("list", B)


# ------------------------------------------------------------------
# Softmax helper
# ------------------------------------------------------------------

softmax <- function(eta) {
  eta <- eta - apply(eta, 1, max)
  exp_eta <- exp(eta)
  exp_eta / rowSums(exp_eta)
}


# ------------------------------------------------------------------
# Monte Carlo loop
# ------------------------------------------------------------------

for (b in seq_len(B)) {
  
  cat("Replication:", b, "/", B, "\n")
  
  # ---------------------------------------------------------------
  # 1. Covariates
  # ---------------------------------------------------------------
  
  X1 <- rnorm(n)
  X2 <- rnorm(n)
  X3 <- rnorm(n)
  
  Z <- cbind(
    X1 = X1,
    X2 = X2,
    X3 = X3
  )
  
  
  # ---------------------------------------------------------------
  # 2. True multinomial GPS
  # ---------------------------------------------------------------
  
  eta_A <-  0.5 * X1 - 0.3 * X2
  eta_B <- -0.3 * X1 + 0.5 * X3
  eta_C <- rep(0, n)
  
  true_ps <- softmax(
    cbind(
      A = eta_A,
      B = eta_B,
      C = eta_C
    )
  )
  
  
  # ---------------------------------------------------------------
  # 3. Generate treatment
  # ---------------------------------------------------------------
  
  Tr <- vapply(
    seq_len(n),
    function(i) {
      sample(
        c("A", "B", "C"),
        size = 1,
        prob = true_ps[i, ]
      )
    },
    character(1)
  )
  
  Tr <- factor(
    Tr,
    levels = c("A", "B", "C")
  )
  
  
  # ---------------------------------------------------------------
  # 4. Potential-outcome model
  # ---------------------------------------------------------------
  
  epsilon_y <- rnorm(n)
  
  baseline <-
    1 +
    0.5 * X1 -
    0.3 * X2 +
    0.2 * X3 +
    epsilon_y
  
  Y_A <- baseline
  Y_B <- baseline + 1
  Y_C <- baseline + 2
  
  # Observed outcome
  Y <- ifelse(
    Tr == "A", Y_A,
    ifelse(Tr == "B", Y_B, Y_C)
  )
  
  
  # ---------------------------------------------------------------
  # 5. Fit M-LBCNet
  #
  # Keep this deliberately small for first simulation.
  # ---------------------------------------------------------------
  
  fit <- m_lbcnet(
    Z = Z,
    Tr = Tr,
    Y = Y,
    
    K = 19,
    
    hidden_dim = 8,
    num_hidden_layers = 0,
    
    vae_epochs = 0,
    
    max_epochs = 5000,
    rolling_window = 1,
    lsd_threshold = 5,
    
    seed = 20,
    
    show_progress = FALSE,
    compute_variance = TRUE
  )
  
  
  # ---------------------------------------------------------------
  # 6. Treatment-specific means
  # ---------------------------------------------------------------
  
  temp_mu <- fit$means
  
  temp_mu$true <- true_mu[
    as.character(temp_mu$treatment)
  ]
  
  temp_mu$bias <- temp_mu$estimate - temp_mu$true
  
  temp_mu$covered <-
    temp_mu$ci_lower <= temp_mu$true &
    temp_mu$ci_upper >= temp_mu$true
  
  temp_mu$rep <- b
  
  mean_results[[b]] <- temp_mu
  
  
  # ---------------------------------------------------------------
  # 7. Pairwise ATEs
  # ---------------------------------------------------------------
  
  temp_ate <- fit$pairwise_ate
  
  temp_ate$contrast <- paste0(
    temp_ate$treatment_1,
    "-",
    temp_ate$treatment_2
  )
  
  temp_ate$true <- true_ate[
    temp_ate$contrast
  ]
  
  temp_ate$bias <-
    temp_ate$estimate -
    temp_ate$true
  
  temp_ate$covered <-
    temp_ate$ci_lower <= temp_ate$true &
    temp_ate$ci_upper >= temp_ate$true
  
  temp_ate$rep <- b
  
  ate_results[[b]] <- temp_ate
}


# ------------------------------------------------------------------
# Combine results
# ------------------------------------------------------------------

mean_results <- do.call(
  rbind,
  mean_results
)

ate_results <- do.call(
  rbind,
  ate_results
)


# ------------------------------------------------------------------
# Summary for treatment-specific means
# ------------------------------------------------------------------

mean_summary <- do.call(
  rbind,
  lapply(
    split(
      mean_results,
      mean_results$treatment
    ),
    function(x) {
      
      data.frame(
        treatment = x$treatment[1],
        
        true = x$true[1],
        
        mean_estimate =
          mean(x$estimate),
        
        bias =
          mean(x$estimate - x$true),
        
        empirical_sd =
          sd(x$estimate),
        
        mean_se =
          mean(x$se),
        
        coverage =
          mean(x$covered)
      )
    }
  )
)

rownames(mean_summary) <- NULL

print(mean_summary)


# ------------------------------------------------------------------
# Summary for pairwise ATEs
# ------------------------------------------------------------------

ate_summary <- do.call(
  rbind,
  lapply(
    split(
      ate_results,
      ate_results$contrast
    ),
    function(x) {
      
      data.frame(
        contrast = x$contrast[1],
        
        true = x$true[1],
        
        mean_estimate =
          mean(x$estimate),
        
        bias =
          mean(x$estimate - x$true),
        
        empirical_sd =
          sd(x$estimate),
        
        mean_se =
          mean(x$se),
        
        coverage =
          mean(x$covered)
      )
    }
  )
)

rownames(ate_summary) <- NULL

print(ate_summary)