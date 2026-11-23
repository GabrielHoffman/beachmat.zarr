# beachmat.zarr

[beachmat](https://bioconductor.org/packages/beachmat) bindings for Zarr-backed matrices, the Zarr counterpart of
[beachmat.hdf5](https://bioconductor.org/packages/beachmat.hdf5).
`initializeCpp()` turns a [ZarrArray](https://bioconductor.org/packages/ZarrArray) seed into a tatami pointer that C++ code reads directly from disk.

```r
library(ZarrArray)
library(beachmat.zarr)

# Sparse AnnData matrix (group with data/indices/indptr)
X <- ZarrSparseMatrix("data.zarr", "X")    # genes x cells, like H5ADMatrix
ptr <- initializeCpp(X)
cell_totals <- beachmat::tatami.column.sums(ptr, num.threads = 8)

# Dense 2D array
ptr2 <- initializeCpp(ZarrArray("dense.zarr"))
```

## Why

HDF5 serializes all reads behind a global lock, so multi-threaded tatami code gains little.
Zarr chunks are independent files, and every extractor here has its own reader and decode buffers,
so chunk reads and decompression run in parallel.

## Supported

| | |
|---|---|
| Formats | Zarr v2 (`.zarray`) and v3 (`zarr.json`), local filesystem |
| Matrices | dense 2D arrays; sparse CSR/CSC groups in the AnnData layout (`data`/`indices`/`indptr`, incl. `subdata`) |
| Types | int8–64, uint8–64, float32/64, bool; little or big endian |
| Layout | C or F order (v3 `transpose` codec), v2 `dimension_separator`, v3 `default`/`v2` chunk keys, missing chunks → `fill_value` |
| Codecs | none, zlib, gzip, crc32c, plus blosc, zstd and lz4 if found at install time (`zarrCodecs()`) |

Anything else (sharding, v2 filters, bz2/lzma, remote/S3 stores, strings) falls back automatically to
beachmat's generic block-processing method, with a message (silence it with `initializeOptions("fallback.message", FALSE)`).

As with HDF5, sparse access is fast along the compressed dimension (rows for CSR, columns for CSC) and
needs a full scan along the other.

## Installation

Needs zlib. The `configure` script finds c-blosc, zstd and lz4 with `pkg-config`, or under
`/opt/homebrew`, `/usr/local` or `/usr` (on macOS: `brew install c-blosc zstd lz4`). To set flags by hand,
use `BEACHMAT_ZARR_CPPFLAGS` and `BEACHMAT_ZARR_LIBS`.

## Implementation

- `inst/include/tatami_zarr/zarr_array.hpp`: reads Zarr metadata and codecs, and decodes chunks.
- `DenseMatrix.hpp`: dense arrays via `tatami_chunked::CustomDenseChunkedMatrix`.
- `CompressedSparseMatrix.hpp`, `sparse_*.hpp`: ported from
  [tatami_hdf5](https://github.com/tatami-inc/tatami_hdf5) (MIT, Aaron Lun), with each HDF5 hyperslab read
  replaced by a cached Zarr range read and the HDF5 lock removed.
