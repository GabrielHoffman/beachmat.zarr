#' Options for Zarr matrices
#'
#' Options for initializing Zarr matrices in \code{\link[beachmat.zarr]{initializeCpp}}.
#'
#' @param option String specifying the name of the option.
#' @param value Value of the option.
#'
#' @details
#' The following options are supported:
#' \itemize{
#' \item \code{realize}, a logical scalar specifying whether to load the matrix data from Zarr into memory,
#' and then cache it for future calls with \code{\link[beachmat]{checkMemoryCache}}.
#' This avoids repeated disk I/O and decompression when performing multiple passes through the matrix, at the expense of memory.
#' \item \code{fallback.message}, a logical scalar specifying whether to emit a message when a Zarr array
#' uses a feature that the native reader does not support (e.g., sharding, bz2 compression, remote stores),
#' in which case \code{initializeCpp} falls back to beachmat's generic (slower, block-processing) method.
#' }
#'
#' @return If \code{value} is missing, the current setting of \code{option} is returned.
#' If \code{value} is supplied, it is used to set the option, and the previous value is invisibly returned.
#'
#' @examples
#' initializeOptions("realize")
#' old <- initializeOptions("realize", TRUE)
#' initializeOptions("realize", old)
#'
#' @export
initializeOptions <- function(option, value) {
    old <- get(option, envir=cached.options, inherits=FALSE)
    if (missing(value)) {
        return(old)
    }
    assign(option, value, envir=cached.options)
    invisible(old)
}

cached.options <- new.env()
cached.options$realize <- FALSE
cached.options$fallback.message <- TRUE
