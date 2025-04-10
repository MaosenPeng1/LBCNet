% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/print.R
\name{print.lbc_net}
\alias{print.lbc_net}
\title{Print Basic Information of an lbc_net Object}
\usage{
\method{print}{lbc_net}(x, ...)
}
\arguments{
\item{x}{An object of class "lbc_net", generated by lbc_net().}

\item{...}{Additional arguments (ignored).}
}
\value{
Prints key training details and model parameters.
}
\description{
Provides a concise summary of an lbc_net object, including sample size,
training loss, local balance (LSD) in training process, and key model hyperparameters.
}
\examples{
\dontrun{
model <- lbc_net(data = data, formula = Tr ~ X1 + X2 + X3 + X4)
print(model)  # Displays a concise overview
}

}
\seealso{
\code{\link{summary}}, \code{\link{getLBC}}
}
