#include "Rcpp.h"
#include "Rtatami.h"
#include "tatami_zarr/tatami_zarr.hpp"

#include <string>

//[[Rcpp::export(rng=false)]]
SEXP initialize_from_zarr_dense(std::string path, bool transpose, double cache_size) {
    tatami_zarr::DenseMatrixOptions opt;
    opt.maximum_cache_size = static_cast<std::size_t>(cache_size);
    auto output = Rtatami::new_BoundNumericMatrix();
    output->ptr.reset(new tatami_zarr::DenseMatrix<double, int>(path, transpose, opt));
    return output;
}

//[[Rcpp::export(rng=false)]]
SEXP initialize_from_zarr_sparse(std::string data_path, std::string indices_path, std::string indptr_path, int nrow, int ncol, bool csr, double cache_size) {
    tatami_zarr::CompressedSparseMatrixOptions opt;
    opt.maximum_cache_size = static_cast<std::size_t>(cache_size);
    auto output = Rtatami::new_BoundNumericMatrix();
    output->ptr.reset(new tatami_zarr::CompressedSparseMatrix<double, int>(nrow, ncol, data_path, indices_path, indptr_path, csr, opt));
    return output;
}

// Returns NA if every array in 'paths' can be read natively, otherwise a string explaining why not.
//[[Rcpp::export(rng=false)]]
Rcpp::String zarr_unsupported_reason(Rcpp::CharacterVector paths) {
    for (R_xlen_t i = 0; i < paths.size(); ++i) {
        std::string p = Rcpp::as<std::string>(paths[i]);
        try {
            tatami_zarr::read_metadata(p);
        } catch (tatami_zarr::UnsupportedError& e) {
            return Rcpp::String(p + ": " + e.what());
        }
    }
    return Rcpp::String(NA_STRING);
}

//[[Rcpp::export(rng=false)]]
Rcpp::LogicalVector zarr_codecs_available() {
    Rcpp::LogicalVector out = Rcpp::LogicalVector::create(
        Rcpp::Named("zlib") = true,
        Rcpp::Named("gzip") = true,
#ifdef TATAMI_ZARR_HAS_BLOSC
        Rcpp::Named("blosc") = true,
#else
        Rcpp::Named("blosc") = false,
#endif
#ifdef TATAMI_ZARR_HAS_ZSTD
        Rcpp::Named("zstd") = true,
#else
        Rcpp::Named("zstd") = false,
#endif
#ifdef TATAMI_ZARR_HAS_LZ4
        Rcpp::Named("lz4") = true
#else
        Rcpp::Named("lz4") = false
#endif
    );
    return out;
}
