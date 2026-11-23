# library(testthat); library(beachmat.zarr); source("helper-zarr.R"); source("test-sparse.R")

set.seed(1)
y <- Matrix::rsparsematrix(53, 31, 0.15)
y[5, ] <- 0          # an empty row
y[, 9] <- 0          # an empty column
y <- as(y, "CsparseMatrix")

for (version in c(2L, 3L)) {
    for (layout in c("CSC", "CSR")) {
        for (cname in names(test_compressors())) {
            test_that(sprintf("sparse v%d %s %s matches", version, layout, cname), {
                seed <- write_sparse_zarr(y, layout=layout, version=version, compressor=test_compressors()[[cname]])
                ptr <- initializeCpp(seed)
                expect_tatami_matches(ptr, y)
                expect_true(beachmat::tatami.is.sparse(ptr))
                expect_identical(beachmat::tatami.prefer.rows(ptr), layout == "CSR")
            })
        }
    }
}

test_that("sparse DelayedMatrix wrappers and transposition work", {
    seed <- write_sparse_zarr(y, layout="CSR")
    z <- DelayedArray::DelayedArray(seed)
    expect_tatami_matches(initializeCpp(z), y)
    expect_tatami_matches(initializeCpp(BiocGenerics::t(z)), Matrix::t(y))
    expect_tatami_matches(initializeCpp(z[3:40, 2:20]), y[3:40, 2:20])
})

test_that("sparse integer and narrow types work", {
    y2 <- abs(round(y * 1000))
    seed <- write_sparse_zarr(y2, layout="CSC", value_type="integer")
    expect_tatami_matches(initializeCpp(seed), y2)
})

test_that("tiny cache sizes still give correct results", {
    seed <- write_sparse_zarr(y, layout="CSR", compressor=Rarr::use_zlib())
    expect_tatami_matches(initializeCpp(seed, zarr.cache.size=0), y, threads=3)
})

test_that("realization into memory works for sparse matrices", {
    seed <- write_sparse_zarr(y, layout="CSC")
    ptr1 <- initializeCpp(seed, zarr.realize=TRUE)
    ptr2 <- initializeCpp(seed, zarr.realize=TRUE)
    expect_identical(capture.output(print(ptr1)), capture.output(print(ptr2)))
    expect_tatami_matches(ptr1, y)
})

test_that("unsupported codecs fall back to the generic method", {
    seed <- write_sparse_zarr(y, layout="CSC", compressor=Rarr::use_bz2())
    expect_message(ptr <- initializeCpp(seed), "fallback")
    expect_tatami_matches(ptr, y, threads=1)
})

test_that("AnnData attributes are honoured when dim/layout are not supplied", {
    # AnnData stores cells x genes as CSR; ZarrArray presents it transposed (genes x cells) as CSC.
    store <- tempfile(fileext=".zarr")
    Rarr::write_zarr_group(store, "X", zarr_version=2L)
    comp <- as(Matrix::t(y), "CsparseMatrix") # CSR of y
    write_1d(comp@x, file.path(store, "X", "data"), 11L, "double", Rarr::use_zlib(), 2L)
    write_1d(as.integer(comp@i), file.path(store, "X", "indices"), 11L, "integer", Rarr::use_zlib(), 2L)
    write_1d(as.integer(comp@p), file.path(store, "X", "indptr"), 11L, "integer", Rarr::use_zlib(), 2L)
    Rarr::write_zarr_attributes(file.path(store, "X"),
        new.zattrs=list(`encoding-type`="csr_matrix", `encoding-version`="0.1.0", shape=dim(y)))

    seed <- ZarrSparseMatrixSeed(store, "X")
    ref <- as(DelayedArray::DelayedArray(seed), "dgCMatrix")
    expect_equal(unname(as.matrix(ref)), unname(as.matrix(Matrix::t(y))))
    expect_tatami_matches(initializeCpp(seed), ref)
})
