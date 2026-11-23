# library(testthat); library(beachmat.zarr); source("helper-zarr.R"); source("test-dense.R")

set.seed(2)
y <- matrix(rnorm(47 * 23), 47, 23)

write_dense <- function(x, version=3L, order="F", compressor=NULL, chunk=c(10L, 6L), data_type=storage.mode(x), fill_value=NULL) {
    path <- tempfile(fileext=".zarr")
    Rarr::write_zarr_array(x, path, chunk_dim=chunk, data_type=data_type, order=order,
        compressor=compressor, zarr_version=version, fill_value=fill_value)
    ZarrArraySeed(path)
}

for (version in c(2L, 3L)) {
    for (order in c("C", "F")) {
        for (cname in names(test_compressors())) {
            test_that(sprintf("dense v%d order %s %s matches", version, order, cname), {
                seed <- write_dense(y, version=version, order=order, compressor=test_compressors()[[cname]])
                expect_equal(as.matrix(DelayedArray::DelayedArray(seed)), y) # sanity check on the writer
                ptr <- initializeCpp(seed)
                expect_tatami_matches(ptr, y)
                expect_false(beachmat::tatami.is.sparse(ptr))
            })
        }
    }
}

test_that("dense chunk shapes larger than the array work", {
    seed <- write_dense(y, chunk=c(100L, 100L))
    expect_tatami_matches(initializeCpp(seed), y)
})

test_that("dense integer and logical arrays work", {
    yi <- matrix(sample(-1000:1000, 47 * 23, replace=TRUE), 47, 23)
    expect_tatami_matches(initializeCpp(write_dense(yi)), yi)

    yl <- matrix(runif(47 * 23) > 0.5, 47, 23)
    expect_tatami_matches(initializeCpp(write_dense(yl)), yl + 0)
})

test_that("missing chunks are filled with the fill value", {
    y0 <- y
    y0[1:10, ] <- 0 # whole chunk-row of zeros; Rarr skips writing chunks equal to the fill value.
    seed <- write_dense(y0, fill_value=0)
    expect_tatami_matches(initializeCpp(seed), y0)
})

test_that("dense DelayedMatrix operations work", {
    z <- DelayedArray::DelayedArray(write_dense(y, compressor=Rarr::use_zlib()))
    expect_tatami_matches(initializeCpp(z), y)
    expect_tatami_matches(initializeCpp(BiocGenerics::t(z)), base::t(y))
    expect_tatami_matches(initializeCpp(z[5:30, c(1, 3, 20)]), y[5:30, c(1, 3, 20)])
})

test_that("tiny cache sizes still give correct results for dense arrays", {
    seed <- write_dense(y, compressor=Rarr::use_gzip())
    expect_tatami_matches(initializeCpp(seed, zarr.cache.size=0), y, threads=3)
})

test_that("unsupported dense arrays fall back", {
    seed <- write_dense(y, compressor=Rarr::use_lzma())
    expect_message(ptr <- initializeCpp(seed), "fallback")
    expect_tatami_matches(ptr, y, threads=1)
})

test_that("writeZarrArray output works", {
    path <- tempfile(fileext=".zarr")
    z <- writeZarrArray(y, path, chunkdim=c(8L, 5L))
    expect_tatami_matches(initializeCpp(z), y)
})
