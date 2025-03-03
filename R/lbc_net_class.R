#' Class "lbc_net" of Fitted LBC-Net Models
#'
#' @description
#' An object of class `lbc_net` represents a fitted LBC-Net model, including estimated propensity scores,
#' local balance metrics, and model parameters.
#'
#' @details
#' The `lbc_net` class provides methods for extracting fitted values, evaluating balance, and summarizing model performance.
#' Below are the available S3 methods for objects of class `lbc_net`:
#'
#' @section Usage:
#' \preformatted{
#' ## S3 method for class 'lbc_net'
#' est_effect(object, Y, ...)
#'
#' ## S3 method for class 'lbc_net'
#' getLBC(object, names, ...)
#'
#' ## S3 method for class 'lbc_net'
#' print(object, ...)
#'
#' ## S3 method for class 'lbc_net'
#' summary(object, ...)
#' }
#'
#' The functions `gsd()` and `lsd()` are standard (non-S3) functions and should be called directly:
#' \preformatted{
#' gsd(object, ...)
#' lsd(object, ...)
#' }
#'
#' These methods allow users to efficiently analyze and interpret the results of the LBC-Net model for causal inference.
#'
#' @seealso
#' \describe{
#'   \item{\code{\link{lbc_net}}}{for model fitting.}
#'   \item{\code{\link{getLBC}}}{for extracting components.}
#'   \item{\code{\link{lsd}} and \code{\link{gsd}}}{for balance assessment.}
#'   \item{\code{\link[=plot.lsd]{plot.lsd}}}{for visualizing local balance results.}
#' }
#'
#' @importFrom methods setClass
#' @export
methods::setClass("lbc_net")

