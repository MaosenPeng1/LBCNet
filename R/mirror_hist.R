#' @importFrom utils globalVariables
utils::globalVariables(c("count"))

#' Mirror Histogram of Propensity Scores
#'
#' @description Creates a mirror histogram to compare the distribution of propensity scores
#' between treated and control groups introduced by Li and Greene (2013).
#'
#' @param object An optional object of class `lbc_net`. If provided, extracts `ps` (fitted propensity scores)
#' and `Tr` (treatment assignment).
#' @param ps A numeric vector of propensity scores. Required if `object` is not provided.
#' @param Tr A binary numeric vector indicating treatment assignment (1 for treatment, 0 for control).
#' Required if `object` is not provided.
#' @param bins Integer specifying the number of bins in the histogram. Default is 70.
#' @param size Numeric specifying the line size for the histogram bars. Default is 0.5.
#' @param theme.size Numeric specifying the base font size for the theme. Default is `15`.
#' @param grid Logical indicating whether to include gridlines in the plot background. Default is `TRUE`.
#'
#' @param ... Additional arguments passed to `ggplot2` layers for customization.
#'
#' @details
#' This function creates a mirror histogram where the control group (Z=0) is displayed above the x-axis
#' and the treatment group (Z=1) is displayed below the x-axis, making it easier to compare the distribution of
#' propensity scores across groups.
#'
#' @return A `ggplot2` object for further customization or direct display.
#'
#' @examples
#' # Example with manually provided propensity scores and treatment indicators
#' set.seed(123)
#' ps <- runif(10000)  # Simulated propensity scores
#' Tr <- sample(0:1, 10000, replace = TRUE)  # Random treatment assignment
#' mirror_hist(ps = ps, Tr = Tr, bins = 50, size = 0.8)
#'
#' \dontrun{
#' # Example with an `lbc_net` object
#' model <- lbc_net(data = data, formula = Tr ~ X1 + X2 + X3 + X4)
#' mirror_hist(model)
#' }
#'
#' @importFrom ggplot2 ggplot aes geom_histogram after_stat geom_hline labs
#' @importFrom ggplot2 theme element_blank element_line element_text
#' @importFrom dplyr filter
#' @export
mirror_hist <- function(object = NULL, ps = NULL, Tr = NULL, bins = 70, size = 0.5, 
                        theme.size = 15, grid = TRUE, ...) {
  
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
  
  data <- data.frame(ps = ps, Z = Tr)
  
  plot <- ggplot2::ggplot(data, ggplot2::aes(x = ps)) +
    ggplot2::geom_histogram(ggplot2::aes(y = after_stat(count)), fill = "white", color = 'black',
                            data = ~ dplyr::filter(., Z == 0), bins = bins, size = size, ...) +
    ggplot2::geom_histogram(ggplot2::aes(y = -after_stat(count)), fill = "white", color = 'black',
                            data = ~ dplyr::filter(., Z == 1), bins = bins, size = size, ...) +
    ggplot2::geom_hline(yintercept = 0) +
    ggplot2::labs(x = "Propensity Score", y = "Frequency") +
    ggplot2::theme(panel.background = ggplot2::element_blank(),
                   axis.line = ggplot2::element_line(colour = "black"),
                   axis.title = ggplot2::element_text(size = theme.size),
                   axis.text = ggplot2::element_text(size = theme.size),
                   plot.title = ggplot2::element_text(size = theme.size),
                   legend.text = ggplot2::element_text(size = theme.size),
                   legend.title = ggplot2::element_text(size = theme.size))
  
  if (grid) {
    plot <- plot + ggplot2::theme(panel.grid.major = ggplot2::element_line(color = "grey90"),
                                  panel.grid.minor = ggplot2::element_line(color = "grey95"))
  }
  
  return(plot)
}

