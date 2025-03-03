#' @importFrom utils globalVariables
utils::globalVariables(c("Z", "avg_ps", "prop_Z"))

#' Calibration Plot for Propensity Scores
#'
#' @description Creates a calibration plot to assess the local calibration of estimated propensity scores.
#'
#' @param object An optional object of class `lbc_net`.
#' @param ps A numeric vector of propensity scores. Required if `object` is not provided.
#' @param Tr A binary numeric vector indicating treatment assignment (1 for treatment, 0 for control).
#' Required if `object` is not provided.
#' @param breaks Integer specifying the number of bins to divide the propensity scores into. Default is 10.
#' @param theme.size Numeric specifying the base font size for the theme. Default is `15`.
#' @param ref.color Character specifying the color of the reference line. Default is "red".
#' @param ... Additional arguments passed to `ggplot2` layers for customization.
#'
#' @details
#' This plot assesses the local calibration of model-based propensity score estimation methods.
#' The estimated propensity scores are divided into `breaks` equal-length intervals, and the
#' average propensity scores and treatment proportions are calculated for each subgroup and plotted.
#' The dashed red line represents the 45-degree reference line, indicating perfect calibration.
#'
#' A well-calibrated model should align closely with this line, ensuring that the estimated propensity
#' scores match the observed treatment proportions within each bin. Deviations suggest poor calibration.
#'
#' @return A `ggplot2` object for further customization or direct display.
#'
#' @examples
#' # Example with manually provided propensity scores and treatment indicators
#' set.seed(123)
#' ps <- runif(100)  # Simulated propensity scores
#' Tr <- sample(0:1, 100, replace = TRUE)  # Random treatment assignment
#' plot_calib(ps = ps, Tr = Tr, breaks = 10)
#'
#' \dontrun{
#' # Example with an `lbc_net` object
#' model <- lbc_net(data = data, formula = Tr ~ X1 + X2 + X3 + X4)
#' plot_calib(model)
#' }
#'
#' @importFrom ggplot2 ggplot aes geom_point geom_line geom_abline labs
#' @importFrom ggplot2 theme_bw theme element_blank element_text
#' @importFrom dplyr group_by summarise %>%
#' @export
plot_calib <- function(object = NULL, ps = NULL, Tr = NULL, breaks = 10, theme.size = 15, ref.color = "red", ...) {

  if (!is.null(object)) {
    if (!inherits(object, "lbc_net")) {
      stop("Error: `object` must be of class 'lbc_net'.")
    }
    ps <- getLBC(object, "fitted.values")
    Tr <- getLBC(object, "Tr")
  }

  if (is.null(ps) || is.null(Tr)) {
    stop("Error: Must provide either an `lbc_net` object or both `ps` and `Tr`.")
  }

  ps_group <- cut(ps, breaks = breaks, include.lowest = TRUE)
  group_data <- dplyr::group_by(data.frame(ps = ps, Z = Tr, ps_group), ps_group) %>%
    dplyr::summarise(avg_ps = mean(ps), prop_Z = mean(Z), .groups = 'drop')

  p <- ggplot2::ggplot(group_data, ggplot2::aes(x = avg_ps, y = prop_Z)) +
    ggplot2::geom_point(...) +
    ggplot2::geom_line(...) +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = ref.color) +
    ggplot2::labs(x = "Average Estimated Propensity Score", y = "Observed Proportion of Z = 1") +
    ggplot2::theme_bw(base_size = theme.size) +
    ggplot2::theme(panel.grid.major = ggplot2::element_blank(),
                   panel.grid.minor = ggplot2::element_blank(),
                   axis.title = ggplot2::element_text(size = theme.size),
                   axis.text = ggplot2::element_text(size = theme.size),
                   text = ggplot2::element_text(size = theme.size))

  return(p)
}
