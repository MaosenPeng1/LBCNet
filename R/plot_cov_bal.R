#' @importFrom utils globalVariables
utils::globalVariables(c("covariate_value"))

#' Plot Covariate Distribution Between Groups
#'
#' @description
#' Plots the distribution of a specified covariate in the treated and control groups.
#' Allows visualization of weighted or unweighted distributions using histogram or density plots.
#'
#' @param object Optional. An object of class `lbc_net`. If provided, extracts `Z`, `Tr`, and weights.
#' @param Z A data frame or matrix of covariates. Required if `object` is not provided.
#' @param Tr A binary numeric vector for treatment assignment (1 = treated, 0 = control).
#' Required if `object` is not provided.
#' @param wt A numeric vector of weights. If NULL and `object` is provided, weights are extracted from `object`.
#' Defaults to `rep(1, length(Tr))` if not provided.
#' @param use_weights Logical. If `TRUE` (default), applies weights to the distributions.
#' If `FALSE`, shows the unweighted covariate distribution, even when an `lbc_net` object is provided.
#' @param cov A character string specifying the covariate name to plot. Defaults to the first column of `Z`.
#' @param plot_type Character string: either `"hist"` (histogram) or `"density"` (default is `"density"`).
#' @param color_treated Color for the treated group. Default is `"red"`.
#' @param color_control Color for the control group. Default is `"blue"`.
#' @param bins Number of bins for the histogram (if `plot_type = "hist"`). Default is `30`.
#' @param alpha Transparency level for fill. Default is `0.4`.
#' @param theme.size Base font size for plot theme. Default is `15`.
#' @param suppress Logical. If `TRUE` (default), suppresses warnings generated during plot rendering (e.g., bandwidth selection warnings from `density()`). 
#' If `FALSE`, warnings will be displayed as usual.
#' @param ... Additional arguments passed to `ggplot2` geoms.
#'
#' @return A `ggplot2` object showing the covariate distribution.
#'
#' @examples
#' \dontrun{
#' # Using an lbc_net object
#' plot_cov_bal(model, cov = "X1")
#'
#' # Manual input
#' plot_cov_bal(Z = Z, Tr = Tr, wt = weights, cov = "X1")
#' }
#'
#' @importFrom ggplot2 ggplot aes geom_density geom_histogram labs theme element_blank
#' @importFrom ggplot2 element_line element_text
#' @importFrom dplyr filter
#' @export
plot_cov_bal <- function(object = NULL,
                                Z = NULL,
                                Tr = NULL,
                                wt = NULL,
                                use_weights = TRUE,
                                cov = NULL,
                                plot_type = c("density", "hist"),
                                color_treated = "red",
                                color_control = "blue",
                                bins = 30,
                                alpha = 0.4,
                                theme.size = 15,
                                suppress = TRUE,
                                ...) {
  
  # Argument matching for plot_type
  plot_type <- match.arg(plot_type)
  
  # Extract from lbc_net object if provided
  if (!is.null(object)) {
    if (!inherits(object, "lbc_net")) {
      stop("Error: `object` must be of class 'lbc_net'.")
    }
    Z <- getLBC(object, "Z")
    Tr <- getLBC(object, "Tr")
    
    if (use_weights && is.null(wt)) {
      wt <- getLBC(object, "weights")
    }
  }
  
  # Checks for manual input
  if (is.null(Z) || is.null(Tr)) {
    stop("Must provide either `object` or both `Z` and `Tr`.")
  }
  
  # Set weights if NULL (unweighted)
  if (!use_weights || is.null(wt)) {
    wt <- rep(1, length(Tr))
  }
  
  # If covariate not specified, pick the first one
  if (is.null(cov)) {
    cov <- colnames(Z)[1]
  }
  
  # Check if the covariate exists in Z
  if (!cov %in% colnames(Z)) {
    stop(paste0("Covariate '", cov, "' not found in Z."))
  }
  
  # Create data frame for plotting
  plot_data <- data.frame(
    covariate_value = Z[, cov],
    Tr = as.factor(Tr),
    wt = wt
  )
  
  is_categorical <- is.factor(plot_data$covariate_value) || length(unique(plot_data$covariate_value)) <= 5
  
  if (is_categorical) {
    plot_data$covariate_value <- factor(plot_data$covariate_value)
    plot_type <- "hist"
  }
  
  # Create the plot base
  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = covariate_value, weight = wt, fill = Tr, color = Tr))
  
  # Add the appropriate plot layer
  if (plot_type == "density" && !is_categorical) {
    p <- p +
      ggplot2::geom_density(alpha = alpha, ...) +
      ggplot2::scale_fill_manual(values = c("0" = color_control, "1" = color_treated)) +
      ggplot2::scale_color_manual(values = c("0" = color_control, "1" = color_treated))
  } else {
    if (is_categorical) {
      p <- p +
        ggplot2::geom_bar(alpha = alpha, position = "dodge", ...) +
        ggplot2::scale_x_discrete(drop = FALSE)
    } else {
      p <- p +
        ggplot2::geom_histogram(alpha = alpha, bins = bins, position = "identity", ...)
    }
    
    p <- p +
      ggplot2::scale_fill_manual(values = c("0" = color_control, "1" = color_treated)) +
      ggplot2::scale_color_manual(values = c("0" = color_control, "1" = color_treated))
  }
  
  # Finalize the plot with labels and theme
  p <- p +
    ggplot2::labs(
      x = cov,
      y = ifelse(plot_type == "density", "Density", "Frequency"),
      fill = "Group",
      color = "Group"
    ) +
    ggplot2::theme(
      panel.background = ggplot2::element_blank(),
      axis.line = ggplot2::element_line(color = "black"),
      axis.title = ggplot2::element_text(size = theme.size),
      axis.text = ggplot2::element_text(size = theme.size),
      legend.title = ggplot2::element_text(size = theme.size),
      legend.text = ggplot2::element_text(size = theme.size)
    )
  
  if (suppress) {
    suppressWarnings(print(p))
  } else {
    print(p)
  }
  invisible(p)
}
