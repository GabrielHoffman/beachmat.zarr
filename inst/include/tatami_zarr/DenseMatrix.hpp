#ifndef TATAMI_ZARR_DENSE_MATRIX_HPP
#define TATAMI_ZARR_DENSE_MATRIX_HPP

/**
 * @file DenseMatrix.hpp
 * @brief Dense matrix backed by a 2-dimensional Zarr array.
 *
 * Chunk caching and all extraction modes (full/block/index, oracular or not)
 * are delegated to `tatami_chunked::CustomDenseChunkedMatrix`; this file only
 * supplies the per-chunk decoding. Each extractor owns its own `ChunkReader`,
 * so extraction is fully parallel across threads.
 */

#include "zarr_array.hpp"
#include "tatami/tatami.hpp"
#include "tatami_chunked/tatami_chunked.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tatami_zarr {

namespace DenseMatrix_internal {

template<typename Index_>
class Workspace final : public tatami_chunked::CustomDenseChunkedMatrixWorkspace<double, Index_> {
public:
    Workspace(std::shared_ptr<const ArrayMetadata> meta, bool transpose) :
        my_meta(meta), my_reader(std::move(meta)), my_transpose(transpose) {}

private:
    std::shared_ptr<const ArrayMetadata> my_meta;
    ChunkReader my_reader;
    bool my_transpose; // tatami rows correspond to Zarr dim 1 if true.

    std::vector<unsigned char> my_raw;
    std::vector<double> my_chunk;
    std::vector<zsize_t> my_coords{0, 0};
    Index_ my_last_r = -1, my_last_c = -1;

    // Strides (in elements) for tatami row/column within a decoded chunk.
    std::size_t my_rstride = 0, my_cstride = 0;

    const double* fetch(Index_ chunk_r, Index_ chunk_c) {
        if (chunk_r != my_last_r || chunk_c != my_last_c) {
            // Zarr dimension 0/1 chunk coordinates.
            my_coords[0] = my_transpose ? chunk_c : chunk_r;
            my_coords[1] = my_transpose ? chunk_r : chunk_c;
            my_reader.read(my_coords, my_raw);
            std::size_t n = my_meta->chunk_elements();
            my_chunk.resize(n);
            convert(my_raw.data(), my_meta->dtype, 0, n, my_chunk.data());
            my_last_r = chunk_r;
            my_last_c = chunk_c;

            // Element (z0, z1) of the chunk sits at z0 * c1 + z1 (C) or z0 + z1 * c0 (F).
            std::size_t c0 = my_meta->chunks[0], c1 = my_meta->chunks[1];
            std::size_t s0 = my_meta->fortran_order ? 1 : c1;
            std::size_t s1 = my_meta->fortran_order ? c0 : 1;
            my_rstride = my_transpose ? s1 : s0;
            my_cstride = my_transpose ? s0 : s1;
        }
        return my_chunk.data();
    }

    // value at tatami (target, non_target) within the chunk.
    double at(const double* chunk, bool row, Index_ t, Index_ nt) const {
        std::size_t r = row ? t : nt, c = row ? nt : t;
        return chunk[r * my_rstride + c * my_cstride];
    }

public:
    void extract(Index_ chunk_r, Index_ chunk_c, bool row, Index_ target_start, Index_ target_length,
                 Index_ non_target_start, Index_ non_target_length, double* output, Index_ stride) {
        const double* chunk = fetch(chunk_r, chunk_c);
        for (Index_ p = target_start, end = target_start + target_length; p < end; ++p) {
            double* out = output + static_cast<std::size_t>(p) * stride;
            for (Index_ q = 0; q < non_target_length; ++q) {
                out[q] = at(chunk, row, p, non_target_start + q);
            }
        }
    }

    void extract(Index_ chunk_r, Index_ chunk_c, bool row, Index_ target_start, Index_ target_length,
                 const std::vector<Index_>& non_target_indices, double* output, Index_ stride) {
        const double* chunk = fetch(chunk_r, chunk_c);
        for (Index_ p = target_start, end = target_start + target_length; p < end; ++p) {
            double* out = output + static_cast<std::size_t>(p) * stride;
            for (std::size_t q = 0; q < non_target_indices.size(); ++q) {
                out[q] = at(chunk, row, p, non_target_indices[q]);
            }
        }
    }

    void extract(Index_ chunk_r, Index_ chunk_c, bool row, const std::vector<Index_>& target_indices,
                 Index_ non_target_start, Index_ non_target_length, double* output, Index_ stride) {
        const double* chunk = fetch(chunk_r, chunk_c);
        for (auto p : target_indices) {
            double* out = output + static_cast<std::size_t>(p) * stride;
            for (Index_ q = 0; q < non_target_length; ++q) {
                out[q] = at(chunk, row, p, non_target_start + q);
            }
        }
    }

    void extract(Index_ chunk_r, Index_ chunk_c, bool row, const std::vector<Index_>& target_indices,
                 const std::vector<Index_>& non_target_indices, double* output, Index_ stride) {
        const double* chunk = fetch(chunk_r, chunk_c);
        for (auto p : target_indices) {
            double* out = output + static_cast<std::size_t>(p) * stride;
            for (std::size_t q = 0; q < non_target_indices.size(); ++q) {
                out[q] = at(chunk, row, p, non_target_indices[q]);
            }
        }
    }
};

template<typename Index_>
class Manager final : public tatami_chunked::CustomDenseChunkedMatrixManager<double, Index_> {
public:
    Manager(std::shared_ptr<const ArrayMetadata> meta, bool transpose) : my_meta(std::move(meta)), my_transpose(transpose) {
        if (my_meta->shape.size() != 2) {
            throw std::runtime_error("expected a 2-dimensional Zarr array at '" + my_meta->path + "'");
        }
        int rdim = transpose ? 1 : 0, cdim = transpose ? 0 : 1;
        my_row_stats = tatami_chunked::ChunkDimensionStats<Index_>(
            sanisizer::cast<Index_>(my_meta->shape[rdim]), sanisizer::cast<Index_>(std::min(my_meta->chunks[rdim], std::max<zsize_t>(my_meta->shape[rdim], 1))));
        my_col_stats = tatami_chunked::ChunkDimensionStats<Index_>(
            sanisizer::cast<Index_>(my_meta->shape[cdim]), sanisizer::cast<Index_>(std::min(my_meta->chunks[cdim], std::max<zsize_t>(my_meta->shape[cdim], 1))));
    }

    std::unique_ptr<tatami_chunked::CustomDenseChunkedMatrixWorkspace<double, Index_> > new_workspace() const {
        return std::make_unique<Workspace<Index_> >(my_meta, my_transpose);
    }

    bool prefer_rows() const {
        // Prefer the dimension whose slabs (chunk length x full other extent) are smaller.
        double rslab = static_cast<double>(my_row_stats.chunk_length) * my_col_stats.dimension_extent;
        double cslab = static_cast<double>(my_col_stats.chunk_length) * my_row_stats.dimension_extent;
        return rslab <= cslab;
    }

    const tatami_chunked::ChunkDimensionStats<Index_>& row_stats() const { return my_row_stats; }
    const tatami_chunked::ChunkDimensionStats<Index_>& column_stats() const { return my_col_stats; }

private:
    std::shared_ptr<const ArrayMetadata> my_meta;
    bool my_transpose;
    tatami_chunked::ChunkDimensionStats<Index_> my_row_stats, my_col_stats;
};

}

/**
 * @brief Options for `DenseMatrix`.
 */
struct DenseMatrixOptions {
    /** Size of the slab cache in bytes. */
    std::size_t maximum_cache_size = 100000000;
    /** Whether to always keep at least one slab in the cache. */
    bool require_minimum_cache = true;
};

/**
 * @brief Dense matrix backed by a 2-dimensional Zarr array.
 *
 * By default, tatami rows correspond to the first Zarr dimension (`shape[0]`).
 * Set `transpose = true` to swap this.
 */
template<typename Value_ = double, typename Index_ = int>
class DenseMatrix final : public tatami_chunked::CustomDenseChunkedMatrix<Value_, Index_, double, DenseMatrix_internal::Manager<Index_> > {
    typedef tatami_chunked::CustomDenseChunkedMatrix<Value_, Index_, double, DenseMatrix_internal::Manager<Index_> > Base;

    static tatami_chunked::CustomDenseChunkedMatrixOptions to_options(const DenseMatrixOptions& opt) {
        tatami_chunked::CustomDenseChunkedMatrixOptions out;
        out.maximum_cache_size = opt.maximum_cache_size;
        out.require_minimum_cache = opt.require_minimum_cache;
        return out;
    }

public:
    DenseMatrix(const std::string& path, bool transpose, const DenseMatrixOptions& opt = DenseMatrixOptions()) :
        Base(std::make_shared<DenseMatrix_internal::Manager<Index_> >(read_metadata(path), transpose), to_options(opt)) {}
};

}

#endif
