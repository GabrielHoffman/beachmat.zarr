#' Initialize Zarr-backed matrices
#'
#' Initialize C++ representations of Zarr-backed matrices based on their \pkg{ZarrArray} representations.
#'
#' @param x A \pkg{ZarrArray} seed object, either a \code{ZarrArraySeed} for a dense 2-dimensional array,
#' or a \code{CSR_ZarrSparseMatrixSeed}/\code{CSC_ZarrSparseMatrixSeed} for a sparse matrix in the AnnData layout
#' (a group containing \code{data}, \code{indices} and \code{indptr} arrays).
#' @param zarr.cache.size Number specifying the size of the cache in bytes during data extraction.
#' Larger values reduce repeated chunk decoding during random access, at the cost of increased memory usage.
#' Each extractor (i.e., each thread) has its own cache.
#' @param zarr.realize Logical scalar specifying whether to load the whole matrix into memory, see \code{\link{initializeOptions}}.
#' @param ... Further arguments, passed to the fallback method if the array cannot be read natively.
#'
#' @details
#' The native reader handles local Zarr v2 and v3 arrays with numeric or boolean data types,
#' C or F order, and no compression or blosc, zlib, gzip, zstd, lz4 or crc32c codecs
#' (blosc, zstd and lz4 only if the corresponding libraries were found at install time, see \code{\link{zarrCodecs}}).
#' Missing chunks are filled with the array's fill value.
#'
#' Arrays that use other features (sharding, v2 filters, bz2/lzma, remote stores, non-numeric types)
#' fall back to \pkg{beachmat}'s generic method for arbitrary seeds, which is correct but slower.
#'
#' For sparse matrices, row access is fast for CSR layouts and column access is fast for CSC layouts.
#' Access along the other dimension requires scanning the whole matrix, as with \pkg{beachmat.hdf5}.
#'
#' @return An external pointer that can be used in any \pkg{tatami}-compatible function.
#'
#' @examples
#' library(ZarrArray)
#' y <- matrix(runif(1000), ncol=20, nrow=50)
#' path <- tempfile(fileext=".zarr")
#' z <- writeZarrArray(y, path, chunkdim=c(10, 10))
#' ptr <- initializeCpp(z)
#' beachmat::tatami.dim(ptr)
#'
#' @author Gabriel Hoffman
#' @export
#' @name initializeCpp
#' @importFrom beachmat initializeCpp checkMemoryCache
#' @importFrom DelayedArray getAutoBlockSize
#' @importClassesFrom ZarrArray ZarrArraySeed ZarrSparseMatrixSeed CSR_ZarrSparseMatrixSeed CSC_ZarrSparseMatrixSeed
NULL

.fallback <- function(reason) {
    if (isTRUE(initializeOptions("fallback.message"))) {
        message("beachmat.zarr: using generic fallback (", reason, ")")
    }
}

#' @export
#' @rdname initializeCpp
setMethod("initializeCpp", "ZarrArraySeed", function(x, ..., zarr.cache.size=getAutoBlockSize(), zarr.realize=initializeOptions("realize")) {
    reason <- NA_character_
    if (!is.null(x@s3_client)) {
        reason <- "remote Zarr stores are not supported natively"
    } else if (length(dim(x)) != 2L) {
        reason <- "only 2-dimensional arrays are supported"
    } else if (!(x@type %in% c("double", "integer", "logical"))) {
        reason <- paste0("unsupported type '", x@type, "'")
    } else {
        reason <- zarr_unsupported_reason(x@zarr_path)
    }

    if (!is.na(reason)) {
        .fallback(reason)
        return(callNextMethod(x, ...))
    }

    if (zarr.realize) {
        return(checkMemoryCache("beachmat.zarr", paste("dense", x@zarr_path, sep=":"), function() {
            initializeCpp(as.matrix(DelayedArray::DelayedArray(x)))
        }))
    }

    initialize_from_zarr_dense(x@zarr_path, transpose=FALSE, cache_size=as.double(zarr.cache.size))
})

.sparse_paths <- function(x) {
    base <- file.path(x@zarr_store, sub("^/+", "", x@group))
    base <- sub("/+$", "", base)
    data <- file.path(base, "data")
    if (!is.null(x@subdata)) {
        data <- file.path(data, x@subdata)
    }
    c(data=data, indices=file.path(base, "indices"), indptr=file.path(base, "indptr"))
}

#' @export
#' @rdname initializeCpp
setMethod("initializeCpp", "ZarrSparseMatrixSeed", function(x, ..., zarr.cache.size=getAutoBlockSize(), zarr.realize=initializeOptions("realize")) {
    paths <- .sparse_paths(x)
    reason <- zarr_unsupported_reason(paths)
    if (!is.na(reason)) {
        .fallback(reason)
        return(callNextMethod(x, ...))
    }

    csr <- is(x, "CSR_ZarrSparseMatrixSeed")

    if (zarr.realize) {
        key <- paste("sparse", paths[["data"]], csr, paste(dim(x), collapse="x"), sep=":")
        return(checkMemoryCache("beachmat.zarr", key, function() {
            initializeCpp(as(DelayedArray::DelayedArray(x), "dgCMatrix"))
        }))
    }

    initialize_from_zarr_sparse(
        paths[["data"]], paths[["indices"]], paths[["indptr"]],
        nrow=nrow(x), ncol=ncol(x), csr=csr,
        cache_size=as.double(zarr.cache.size)
    )
})

#' Available Zarr codecs
#'
#' Report which compression codecs this build of \pkg{beachmat.zarr} can decode natively.
#'
#' @return Named logical vector.
#' @examples
#' zarrCodecs()
#' @export
zarrCodecs <- function() {
    zarr_codecs_available()
}
