% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/summary.R
\name{summary.lbc_net}
\alias{summary.lbc_net}
\title{Summary of an lbc_net Object}
\usage{
\method{summary}{lbc_net}(object, Y = NULL, type = "ATE", ...)
}
\arguments{
\item{object}{An object of class `"lbc_net"`, generated by `lbc_net()`.}

\item{Y}{(Optional) A numeric vector of observed outcomes.}

\item{type}{A character string specifying the treatment effect to estimate.
Options:
- `"Y"`: Computes the weighted mean outcome.
- `"ATE"` (default): Computes the Average Treatment Effect.
- `"ATT"`: Computes the Average Treatment Effect on the Treated.}

\item{...}{Additional arguments (ignored).}
}
\value{
A list containing:
\describe{
  \item{\code{sample_info}}{Sample sizes and covariate counts.}
  \item{\code{losses}}{Training losses.}
  \item{\code{local_balance}}{Local standardized differences from training.}
  \item{\code{balance_table}}{Pre- and post-weighting global standardized differences (GSD).}
  \item{\code{treatment_effect}}{Estimated treatment effect, if applicable.}
  \item{\code{gsd}}{GSD after weighting.}
}
}
\description{
Provides a structured summary of an `lbc_net` object,
including loss values, balance assessments, and an optional treatment effect estimate.
}
\details{
The function extracts key model components using \code{\link{getLBC}}.
If `compute_gsd = TRUE`, it calculates global balance using \code{\link{gsd}}.
If `Y` is provided, it estimates treatment effects using \code{\link{est_effect}}.

It is designed for estimating causal effects in settings with continuous or binary outcomes.
For survival outcomes, users should apply appropriate survival analysis models, such as a
weighted Cox model or other time-to-event estimation methods.
}
\examples{
\dontrun{
model <- lbc_net(data = data, formula = Tr ~ X1 + X2 + X3 + X4)
summary(model)  # Summary without treatment effect estimation
summary(model, Y = my_outcome, type = "ATE")  # Summary including treatment effect

out <- summary(model)
names(out)
out$balance_table
}

}
\seealso{
\code{\link{est_effect}}, \code{\link{gsd}}, \code{\link{getLBC}}
}
