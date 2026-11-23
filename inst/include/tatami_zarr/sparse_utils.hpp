// Ported from tatami_hdf5 (https://github.com/tatami-inc/tatami_hdf5, MIT License,
// Copyright (c) Aaron Lun), with HDF5 I/O replaced by Zarr chunk reads.
#ifndef TATAMI_ZARR_SPARSE_UTILS_HPP
#define TATAMI_ZARR_SPARSE_UTILS_HPP


#include <algorithm>
#include <utility>
#include <cstddef>
#include <string>
#include <vector>

#include "sanisizer/sanisizer.hpp"
#include "tatami/tatami.hpp"
#include "zarr_array.hpp"

namespace tatami_zarr {

// Zarr chunk reads use per-extractor file handles and buffers, so no global lock is needed.
template<class Function_>
void serialize(Function_ f) {
    f();
}

namespace CompressedSparseMatrix_internal {

template<typename CachedValue_, typename CachedIndex_>
std::size_t size_of_cached_element(bool needs_cached_value, bool needs_cached_index) {
    return (needs_cached_index ? sizeof(CachedIndex_) : 0) + (needs_cached_value ? sizeof(CachedValue_) : 0);
}

// Per-extractor readers for the 'data' and 'indices' arrays.
// Each keeps a small LRU cache of decoded chunks; as in tatami_hdf5, three chunks are enough
// to hold the partially-read chunks at both ends of a primary element plus one in the middle.
struct Components {
    Components(std::shared_ptr<const ArrayMetadata> value_meta, std::shared_ptr<const ArrayMetadata> index_meta) :
        data_dataset(std::move(value_meta), 3), index_dataset(std::move(index_meta), 3) {}
    Array1DReader data_dataset;
    Array1DReader index_dataset;
};

// In all cases, we know that max_non_zeros can be safely casted between zsize_t and Index_,
// because the value is derived from differences between zsize_t pointers.

template<typename Index_>
struct MatrixDetails {
    MatrixDetails(
        const std::shared_ptr<const ArrayMetadata>& value_meta,
        const std::shared_ptr<const ArrayMetadata>& index_meta,
        Index_ primary_dim, 
        Index_ secondary_dim, 
        const std::vector<zsize_t>& pointers, 
        std::size_t slab_cache_size,
        Index_ max_non_zeros
    ) :
        value_meta(value_meta), 
        index_meta(index_meta), 
        primary_dim(primary_dim), 
        secondary_dim(secondary_dim), 
        pointers(pointers), 
        slab_cache_size(slab_cache_size),
        max_non_zeros(max_non_zeros)
    {}

    const std::shared_ptr<const ArrayMetadata>& value_meta;
    const std::shared_ptr<const ArrayMetadata>& index_meta;

    Index_ primary_dim;
    Index_ secondary_dim;
    const std::vector<zsize_t>& pointers;

    std::size_t slab_cache_size; // size for our own cache of slabs.
    Index_ max_non_zeros;
};

template<typename Index_>
void initialize(const MatrixDetails<Index_>& details, std::unique_ptr<Components>& h5comp) {
    h5comp.reset(new Components(details.value_meta, details.index_meta));
}

inline void destroy(std::unique_ptr<Components>& h5comp) {
    h5comp.reset();
}

// If we're only interested in the block [X, Y), we don't need to extract the entire set of indices for a primary element.
// We only need to extract the first Y elements, as all subsequent elements must have indices greater than or equal to Y.
// Similarly, we only need to extract the last 'dim - X' elements, as all preceding elements must have indices less than X.
// This allows us to narrow the range of indices to be extracted.
template<typename Pointer_, typename Index_>
std::pair<Pointer_, Pointer_> narrow_primary_extraction_range(
    const Pointer_ ptr_start,
    const Pointer_ ptr_end,
    const Index_ block_start,
    const Index_ block_end,
    const Index_ secondary_dim
) {
    const Pointer_ num_nonzeros = ptr_end - ptr_start;
    const Pointer_ new_ptr_end = ptr_start + sanisizer::min(block_end, num_nonzeros);
    const Pointer_ new_ptr_start = ptr_end - sanisizer::min(secondary_dim - block_start, num_nonzeros);

    // It is guaranteed that new_ptr_start <= new_ptr_end.
    // This is because the difference boils down to:
    //
    // -num_nonzeros + num_nonzeros + num_nonzeros, OR
    // -num_nonzeros + block_end + num_nonzeros, OR
    // -num_nonzeros + num_nonzeros + secondary_dim - block_start, OR
    // -num_nonzeros + block_end + secondary_dim - block_start
    // 
    // All of which are guaranteed to be non-negative.
    return std::make_pair(new_ptr_start, new_ptr_end);
}

template<class IndexIt_, typename Index_>
void refine_primary_limits(IndexIt_& indices_start, IndexIt_& indices_end, Index_ extent, Index_ smallest, Index_ largest_plus_one) {
    if (smallest) {
        // Using custom comparator to ensure that we cast to Index_ for signedness-safe comparisons.
        indices_start = std::lower_bound(indices_start, indices_end, smallest, [](Index_ a, Index_ b) -> bool { return a < b; });
    }

    if (largest_plus_one != extent) {
        indices_end = std::lower_bound(indices_start, indices_end, largest_plus_one, [](Index_ a, Index_ b) -> bool { return a < b; });
    }
}

template<typename Slab_, typename Value_, typename Index_>
tatami::SparseRange<Value_, Index_> slab_to_sparse(const Slab_& slab, Value_* value_buffer, Index_* index_buffer) {
    tatami::SparseRange<Value_, Index_> output(slab.number);
    if (slab.value) {
        std::copy_n(slab.value, slab.number, value_buffer);
        output.value = value_buffer;
    }
    if (slab.index) {
        std::copy_n(slab.index, slab.number, index_buffer);
        output.index = index_buffer;
    }
    return output;
}

template<typename Slab_, typename Value_, typename Index_>
Value_* slab_to_dense(const Slab_& slab, Value_* buffer, Index_ extract_length) {
    std::fill_n(buffer, extract_length, 0);
    for (Index_ i = 0; i < slab.number; ++i) {
        buffer[slab.index[i]] = slab.value[i];
    }
    return buffer;
}

}

}

#endif
