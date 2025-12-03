#' @importFrom utils globalVariables
utils::globalVariables(c("x", "y"))

#' Plot Local Standardized Mean Difference (LSD) Results
#'
#' @description Creates plots of Local Standardized Mean Difference (LSD) from an `lsd` object.
#' The function provides visualization options:
#' \itemize{
#'   \item Average LSD over all covariates (default), with an option to include box plots.
#'   \item LSD for a specific covariate by name or column index.
#' }
#'
#' @param x An object of class `lsd` created using \code{\link{lsd}}.
#' @param y Unused, included for compatibility with the \code{\link{plot}} generic.
#' @param cov Character string or numeric index specifying the covariate to plot.
#'   Use `"ALL"` to average over all covariates (default), or specify a covariate name or column index.
#' @param ... Additional parameters for customizing the plot, including but not limited to:
#'   \describe{
#'     \item{`box.loc`}{A numeric vector specifying locations for box plots when `cov = "ALL"`.
#'       Must be a subset of grid points `ck`. Default is `seq(0.1, 0.9, by = 0.2)`. Set to `NULL` to disable box plots.}
#'     \item{`point.color`}{Character string specifying the color of points in the plot. Default is `"#9467bd"`.}
#'     \item{`point.size`}{Numeric specifying the size of points in the plot. Default is `0.8`.}
#'     \item{`line.size`}{Numeric specifying the size of lines in the plot. Default is `0.5`.}
#'     \item{`line.color`}{Character string specifying the color of lines in the plot. Default is `"black"`.}
#'     \item{`theme.size`}{Numeric specifying the base font size for the theme. Default is `15`.}
#'     \item{`boxplot.width`}{Numeric specifying the width of the box plots. Default is `0.02`.}
#'     \item{`outlier.shape`}{Numeric specifying the shape of outliers in box plots. Default is `4`.}
#'     \item{`outlier.size`}{Numeric specifying the size of outliers in box plots. Default is `1`.}
#'   }
#'
#' @details
#' This function provides flexible visualization for LSD results to evaluate the local balance of estimated propensity scores:
#' \itemize{
#'   \item If `cov = "ALL"`, the plot shows the average LSD over all covariates. If `box.loc` is specified, box plots are added to show variability.
#'   \item If a specific covariate is selected using `cov` (either by name or column index), the plot shows LSD for that covariate.
#' }
#'
#' Box plots are only applicable when `cov = "ALL"` and `box.loc` is not `NULL`.
#'
#' @return A `ggplot2` object for further customization or direct display.
#'
#' @seealso
#' \code{\link{lsd}}, \code{\link{getLBC}}, \code{\link[=plot.lsd]{plot.lsd}}
#'
#' @examples
#' \dontrun{
#' # Basic plot using an lsd object
#' plot(lsd_fit)
#'
#' # Plot average LSD across all covariates
#' plot(lsd_result, cov = "ALL")
#'
#' # Plot LSD for a specific covariate
#' plot(lsd_result, cov = 1)
#' plot(lsd_result, cov = "Cov1")
#'
#' # Plot LSD for all covariates with box plots
#' plot(lsd_fit, cov = "ALL", box.loc = seq(0.1, 0.9, by = 0.2))
#'
#' # Customize the plot appearance
#' plot(lsd_fit, cov = "ALL", point.color = "red", point.size = 1, line.size = 1)
#' }
#'
#' @importFrom ggplot2 ggplot aes geom_point geom_line geom_boxplot labs
#' @importFrom ggplot2 theme_bw theme element_blank element_text
#' @importFrom tidyr pivot_longer
#' @importFrom dplyr mutate filter %>%
#' @importFrom tidyselect everything
#'
#' @export
plot.lsd <- function(x, y = NULL, cov = "ALL", ...) {
  
  # Extract additional plot parameters from ...
  args <- list(...)
  box.loc       <- if (!is.null(args$box.loc))       args$box.loc       else seq(0.1, 0.9, by = 0.2)
  point.color   <- if (!is.null(args$point.color))   args$point.color   else "#9467bd"
  point.size    <- if (!is.null(args$point.size))    args$point.size    else 0.8
  line.size     <- if (!is.null(args$line.size))     args$line.size     else 0.5
  line.color    <- if (!is.null(args$line.color))    args$line.color    else "black"
  theme.size    <- if (!is.null(args$theme.size))    args$theme.size    else 15
  boxplot.width <- if (!is.null(args$boxplot.width)) args$boxplot.width else 0.02
  outlier.shape <- if (!is.null(args$outlier.shape)) args$outlier.shape else 4
  outlier.size  <- if (!is.null(args$outlier.size))  args$outlier.size  else 1
  
  object <- x  # Match base R’s plot() generic where 'x' is the first argument
  if (!inherits(object, "lsd")) {
    stop("Error: `object` must be of class 'lsd'.")
  }
  
  LSD  <- object$LSD
  Z    <- object$Z
  ck   <- object$ck
  p_dim <- ncol(Z)
  
  if (cov == "ALL") {
    lsd.colmean <- rowMeans(LSD)
    ds.plot <- data.frame(x = ck, y = lsd.colmean)
    
    p <- ggplot2::ggplot(ds.plot, ggplot2::aes(x = x, y = y)) +
      ggplot2::geom_point(size = point.size, color = point.color) +
      ggplot2::geom_line(linewidth = line.size, color = line.color) +
      ggplot2::theme_bw(base_size = theme.size) +
      ggplot2::theme(
        panel.grid.major = ggplot2::element_blank(),
        panel.grid.minor = ggplot2::element_blank(),
        axis.title       = ggplot2::element_text(size = theme.size),
        axis.text        = ggplot2::element_text(size = theme.size),
        legend.text      = ggplot2::element_text(size = theme.size),
        legend.title     = ggplot2::element_text(size = theme.size)
      ) +
      ggplot2::labs(x = "Propensity Score", y = "LSD(%)")
    
    if (!is.null(box.loc)) {
      ds.box <- as.data.frame(LSD) %>%
        tidyr::pivot_longer(cols = tidyselect::everything(),
                            names_to = "Z", values_to = "y") %>%
        dplyr::mutate(x = rep(ck, each = p_dim)) %>%
        dplyr::filter(sapply(x, function(val) any(dplyr::near(val, box.loc))))
      
      p <- p + ggplot2::geom_boxplot(
        data  = ds.box,
        ggplot2::aes(x = x, y = y, group = x),
        outlier.shape = outlier.shape,
        outlier.size  = outlier.size,
        width         = boxplot.width,
        color         = point.color
      )
    }
    
  } else {
    if (is.numeric(cov)) {
      if (cov < 1 || cov > ncol(LSD)) {
        stop("Error: Column index out of range.")
      }
      lsd.colmean <- LSD[, cov]
    } else {
      if (!cov %in% colnames(Z)) {
        stop("Error: Specified covariate not found in LSD results.")
      }
      lsd.colmean <- LSD[, cov]
    }
    
    ds.plot <- data.frame(x = ck, y = lsd.colmean)
    p <- ggplot2::ggplot(ds.plot, ggplot2::aes(x = x, y = y)) +
      ggplot2::geom_point(size = point.size, color = point.color) +
      ggplot2::geom_line(linewidth = line.size, color = line.color) +
      ggplot2::theme_bw(base_size = theme.size) +
      ggplot2::labs(x = "Propensity Score", y = "LSD(%)")
  }
  
  return(p)
}

