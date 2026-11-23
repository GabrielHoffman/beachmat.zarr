library(Matrix)
library(ZarrArray)

# Compressors to test, as Rarr codec objects (NULL = uncompressed).
# Only the ones this build can decode natively are included.
test_compressors <- function() {
    avail <- beachmat.zarr::zarrCodecs()
    out <- list(none=NULL, zlib=Rarr::use_zlib(), gzip=Rarr::use_gzip())
    if (avail[["blosc"]]) {
        out$blosc_lz4 <- Rarr::use_blosc("lz4")
        out$blosc_zstd_bitshuffle <- Rarr::use_blosc("zstd", shuffle="bitshuffle")
    }
    if (avail[["zstd"]]) out$zstd <- Rarr::use_zstd()
    if (avail[["lz4"]]) out$lz4 <- Rarr::use_lz4()
    out
}

write_1d <- function(x, path, chunk, data_type, compressor, version) {
    Rarr::write_zarr_array(as.array(x), path, chunk_dim=chunk, data_type=data_type,
        compressor=compressor, zarr_version=version)
}

# Writes 'y' (a dgCMatrix) as an AnnData-style sparse group and returns a ZarrSparseMatrixSeed.
write_sparse_zarr <- function(y, layout=c("CSC", "CSR"), version=3L, compressor=NULL,
                              chunk=7L, value_type="double", index_type="integer")
{
    layout <- match.arg(layout)
    store <- tempfile(fileext=".zarr")
    Rarr::write_zarr_group(store, "foo", zarr_version=version)
    group <- file.path(store, "foo")

    if (layout == "CSC") {
        comp <- y
    } else {
        comp <- as(Matrix::t(y), "CsparseMatrix")
    }
    vals <- comp@x
    if (value_type == "integer") vals <- as.integer(vals)
    write_1d(vals, file.path(group, "data"), chunk, value_type, compressor, version)
    write_1d(as.integer(comp@i), file.path(group, "indices"), chunk, index_type, compressor, version)
    write_1d(as.integer(comp@p), file.path(group, "indptr"), chunk, "integer", compressor, version)

    ZarrSparseMatrixSeed(store, "foo", dim=dim(y), sparse.layout=layout)
}

# Compare a tatami pointer against a reference matrix.
expect_tatami_matches <- function(ptr, ref, threads=2) {
    ref <- as.matrix(ref)
    testthat::expect_identical(beachmat::tatami.dim(ptr), dim(ref))
    for (r in unique(c(1L, nrow(ref) %/% 2L, nrow(ref)))) {
        testthat::expect_equal(beachmat::tatami.row(ptr, r), unname(ref[r, ]))
    }
    for (cc in unique(c(1L, ncol(ref) %/% 2L, ncol(ref)))) {
        testthat::expect_equal(beachmat::tatami.column(ptr, cc), unname(ref[, cc]))
    }
    testthat::expect_equal(beachmat::tatami.row.sums(ptr, num.threads=threads), unname(rowSums(ref)))
    testthat::expect_equal(beachmat::tatami.column.sums(ptr, num.threads=threads), unname(colSums(ref)))
    testthat::expect_equal(unname(as.matrix(beachmat::tatami.realize(ptr, num.threads=threads))), unname(ref))
}
