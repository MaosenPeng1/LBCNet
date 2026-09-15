utils::globalVariables(c("center", "lsd", "mean_lsd"))

#' Plot Multi-Treatment Local Standardized Mean Differences
#'
#' @md
#' @description Plots local balance diagnostics from an M-LBCNet fit, with
#'   points and connecting lines at generalized propensity score grid centers.
#'
#' @param x An object of class `m_lbcnet_lsd` returned by
#'   \code{\link{lsd}} for an \code{\link{m_lbcnet}} fit.
#' @param y Unused; included for compatibility with the `plot()` generic.
#' @param type Diagnostic to display. `"versus_population"` (the default)
#'   compares each treatment with the common local ATE population.
#'   `"pairwise"` compares treatment pairs within each localizing treatment's
#'   generalized propensity score neighborhoods.
#' @param cov `"ALL"` (the default) averages LSD over covariates at each
#'   center within each panel. Otherwise, supply one covariate name or a
#'   one-based numeric index in the order of the fitted covariate columns.
#' @param box.loc Numeric vector of centers at which to show boxplots over
#'   covariates when `cov = "ALL"`. Defaults to `seq(0.1, 0.9, by = 0.2)`.
#'   Matching uses [dplyr::near()] to allow floating point differences;
#'   unmatched locations are ignored. Explicit `NULL` disables boxplots.
#'   Ignored when a specific covariate is selected.
#' @param ... Additional arguments, currently unused.
#' @param localizing_treatment Optional treatment label or vector of labels
#'   selecting pairwise facet rows. `NULL` shows all localizing treatments.
#'   Only used when `type = "pairwise"`.
#' @param treatment_pair Optional vector of two distinct treatment labels
#'   selecting a pairwise comparison, in either order. `NULL` shows all pairs.
#'   Only used when `type = "pairwise"`.
#' @param point.color,point.size Point color and size. Defaults are
#'   `"#9467bd"` and `0.8`. The point color also sets the boxplot color.
#' @param line.color,line.size Connecting line color and width. Defaults are
#'   `"black"` and `0.5`.
#' @param theme.size Base font size for [ggplot2::theme_bw()]. Default is `15`.
#' @param boxplot.width Boxplot width in propensity score units. Default is
#'   `0.02`.
#' @param outlier.shape,outlier.size Boxplot outlier shape and size.
#'   Defaults are `4` and `1`; use `outlier.shape = NA` to hide outliers.
#'
#' @details
#' The primary, treatment-versus-population diagnostic has one panel per
#' treatment, in a single row with a shared y-axis scale. In each panel,
#' `center` refers to that treatment's generalized propensity score component.
#'
#' The pairwise diagnostic has one column per treatment pair and one row per
#' localizing treatment, labeled `"Localized by <treatment>"`. For localizing
#' treatment \eqn{r}, both compared treatments use the same neighborhood along
#' \eqn{\pi_r}; the x-axis centers in that row refer to \eqn{\pi_r}.
#' Localizing treatments are never averaged together.
#'
#' With `cov = "ALL"`, lines and points show the mean LSD over valid
#' covariates, and boxplots show the distribution of individual covariate LSD
#' values at selected centers. These are diagnostics for a single dataset,
#' not summaries over simulations or seeds. Selecting a specific covariate
#' shows its LSD directly, without boxplots.
#'
#' Non-finite LSD values or centers, and LSD values at or above the package's
#' unsupported-neighborhood sentinel (\eqn{10^8}), are excluded before
#' summarizing. A warning reports the number of excluded observations in the
#' selected data. An error is raised if no valid observations remain.
#'
#' @return A `ggplot` object, returned without explicitly printing it, which
#'   can be further customized with `ggplot2` layers and themes.
#' @seealso \code{\link{lsd}}, \code{\link{m_lbcnet}},
#'   \code{\link{plot.lsd}}
#' @examples
#' \dontrun{
#' # Assume fit is an m_lbcnet fit with treatments A, B, C and covariate Z1.
#' lsd_fit <- lsd(fit)
#'
#' # Primary treatment-versus-population diagnostic
#' plot(lsd_fit)
#'
#' # Specific covariate, by name or fitted column index
#' plot(lsd_fit, cov = "Z1")
#' plot(lsd_fit, cov = 1)
#'
#' # Full pairwise diagnostic
#' plot(lsd_fit, type = "pairwise")
#'
#' # Pairwise diagnostic localized by treatment A
#' plot(lsd_fit, type = "pairwise", localizing_treatment = "A")
#' plot(lsd_fit, type = "pairwise", treatment_pair = c("A", "B"))
#'
#' # Disable boxplots and customize the returned plot
#' p <- plot(lsd_fit, box.loc = NULL)
#' p + ggplot2::theme(axis.text = ggplot2::element_text(size = 12))
#' }
#' @export
plot.m_lbcnet_lsd <- function(
    x, y = NULL, type = c("versus_population", "pairwise"),
    cov = "ALL", box.loc = seq(0.1, 0.9, by = 0.2), ...,
    localizing_treatment = NULL, treatment_pair = NULL,
    point.color = "#9467bd", point.size = 0.8,
    line.color = "black", line.size = 0.5, theme.size = 15,
    boxplot.width = 0.02, outlier.shape = 4, outlier.size = 1) {
  if (!inherits(x, "m_lbcnet_lsd")) {
    stop("`x` must be of class 'm_lbcnet_lsd'.", call. = FALSE)
  }
  type <- match.arg(type)
  data <- x[[type]]
  facet_columns <- if (type == "versus_population") {
    "treatment"
  } else {
    c("localizing_treatment", "treatment_1", "treatment_2")
  }
  required_columns <- c(facet_columns, "center", "covariate", "lsd")
  if (!is.data.frame(data) || !all(required_columns %in% names(data))) {
    stop("`x$", type, "` must be a data frame containing: ",
         paste(required_columns, collapse = ", "), ".", call. = FALSE)
  }
  if (!is.numeric(data$center) || !is.numeric(data$lsd)) {
    stop("`center` and `lsd` must be numeric columns.", call. = FALSE)
  }
  if (!nrow(data)) {
    stop("No valid observations remain after filtering.", call. = FALSE)
  }

  # Resolve indices before filtering so they always refer to fitted columns.
  covariates <- unique(as.character(data$covariate))
  if (length(cov) != 1L || anyNA(cov) ||
      !(is.character(cov) || is.numeric(cov))) {
    stop("`cov` must be 'ALL', one covariate name, or one numeric index.",
         call. = FALSE)
  }
  all_covariates <- is.character(cov) && cov == "ALL"
  if (!all_covariates) {
    if (is.numeric(cov)) {
      if (!is.finite(cov) || cov != floor(cov) ||
          cov < 1 || cov > length(covariates)) {
        stop("Covariate index out of range: use an integer from 1 to ",
             length(covariates), ".", call. = FALSE)
      }
      cov <- covariates[cov]
    } else if (!cov %in% covariates) {
      stop("Covariate '", cov, "' not found. Available covariates: ",
           paste(covariates, collapse = ", "), ".", call. = FALSE)
    }
    data <- data[data$covariate %in% cov, , drop = FALSE]
  }

  if (type == "pairwise") {
    if (!is.null(localizing_treatment)) {
      available <- unique(as.character(data$localizing_treatment))
      if (!is.atomic(localizing_treatment) || !length(localizing_treatment) ||
          anyNA(localizing_treatment) ||
          !all(localizing_treatment %in% available)) {
        stop("`localizing_treatment` must select existing treatment labels: ",
             paste(available, collapse = ", "), ".", call. = FALSE)
      }
      data <- data[data$localizing_treatment %in% localizing_treatment,
                   , drop = FALSE]
    }
    if (!is.null(treatment_pair)) {
      if (!is.atomic(treatment_pair) || length(treatment_pair) != 2L ||
          anyNA(treatment_pair) || anyDuplicated(treatment_pair)) {
        stop("`treatment_pair` must contain two distinct treatment labels.",
             call. = FALSE)
      }
      selected <-
        (data$treatment_1 == treatment_pair[1L] &
           data$treatment_2 == treatment_pair[2L]) |
        (data$treatment_1 == treatment_pair[2L] &
           data$treatment_2 == treatment_pair[1L])
      if (!any(selected)) {
        stop("Requested `treatment_pair` not found in the pairwise results.",
             call. = FALSE)
      }
      data <- data[selected, , drop = FALSE]
    }
    data$pair <- paste0(data$treatment_1, " vs ", data$treatment_2)
    facet_columns <- c("localizing_treatment", "pair")
  }

  valid <- is.finite(data$lsd) & is.finite(data$center) &
    data$lsd < .m_lbcnet_lsd_invalid_value
  excluded <- sum(!valid)
  if (excluded > 0L) {
    warning("Excluded ", excluded, " observation(s) with non-finite LSD ",
            "or centers, or unsupported LSD (>= ",
            format(.m_lbcnet_lsd_invalid_value, scientific = TRUE), ").",
            call. = FALSE)
  }
  data <- data[valid, , drop = FALSE]
  if (!nrow(data)) {
    stop("No valid observations remain after filtering.", call. = FALSE)
  }

  # Keep the original labels and their order in the diagnostic tables.
  for (column in facet_columns) {
    data[[column]] <- factor(data[[column]], levels = unique(data[[column]]))
  }
  if (all_covariates) {
    plot_data <- stats::aggregate(
      data["lsd"], data[c(facet_columns, "center")], mean
    )
    names(plot_data)[names(plot_data) == "lsd"] <- "mean_lsd"
    mapping <- ggplot2::aes(x = center, y = mean_lsd)
  } else {
    plot_data <- data
    mapping <- ggplot2::aes(x = center, y = lsd)
  }

  p <- ggplot2::ggplot(plot_data, mapping) +
    ggplot2::geom_point(size = point.size, color = point.color) +
    ggplot2::geom_line(linewidth = line.size, color = line.color, group = 1) +
    ggplot2::theme_bw(base_size = theme.size) +
    ggplot2::theme(
      panel.grid.major = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      axis.title = ggplot2::element_text(size = theme.size),
      axis.text = ggplot2::element_text(size = theme.size)
    )
  if (type == "versus_population") {
    p <- p + ggplot2::facet_wrap(~ treatment, nrow = 1) +
      ggplot2::labs(x = "Generalized Propensity Score", y = "LSD (%)")
  } else {
    p <- p + ggplot2::facet_grid(
      localizing_treatment ~ pair,
      labeller = ggplot2::labeller(
        localizing_treatment = function(labels) paste("Localized by", labels)
      )
    ) + ggplot2::labs(
      x = "Local Generalized Propensity Score", y = "Pairwise LSD (%)"
    )
  }

  if (all_covariates && !is.null(box.loc)) {
    if (!is.numeric(box.loc) || any(!is.finite(box.loc))) {
      stop("`box.loc` must be a finite numeric vector or NULL.", call. = FALSE)
    }
    at_box <- vapply(
      data$center, function(value) any(dplyr::near(value, box.loc)), logical(1)
    )
    if (any(at_box)) {
      p <- p + ggplot2::geom_boxplot(
        data = data[at_box, , drop = FALSE],
        mapping = ggplot2::aes(x = center, y = lsd, group = center),
        inherit.aes = FALSE, width = boxplot.width, color = point.color,
        outlier.shape = outlier.shape, outlier.size = outlier.size
      )
    }
  }
  p
}
