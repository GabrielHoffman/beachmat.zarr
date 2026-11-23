#ifndef TATAMI_ZARR_ZARR_ARRAY_HPP
#define TATAMI_ZARR_ZARR_ARRAY_HPP

/**
 * @file zarr_array.hpp
 * @brief Minimal, thread-safe reader for Zarr v2/v3 arrays on a local filesystem.
 *
 * Supported:
 * - Zarr v2 (`.zarray`) and v3 (`zarr.json`) array metadata.
 * - Numeric and boolean data types, either endianness.
 * - C or F chunk order (v2 `order`, v3 `transpose` codec for 1D/2D arrays).
 * - Compressors/codecs: none, blosc, zlib, gzip, zstd, lz4 (numcodecs framing), crc32c (v3, checksum not verified).
 * - v2 `dimension_separator` and v3 `default`/`v2` chunk key encodings.
 * - Missing chunk files are filled with `fill_value`.
 *
 * Not supported (an `UnsupportedError` is thrown): v2 filters, sharding, bz2/lzma,
 * string/structured/extension data types, remote stores.
 *
 * Everything here uses plain file I/O and per-object buffers, so separate reader
 * instances can be used concurrently from different threads.
 */

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include "millijson/millijson.hpp"

#include <zlib.h>
#ifdef TATAMI_ZARR_HAS_BLOSC
#include <blosc.h>
#endif
#ifdef TATAMI_ZARR_HAS_ZSTD
#include <zstd.h>
#endif
#ifdef TATAMI_ZARR_HAS_LZ4
#include <lz4.h>
#endif

namespace tatami_zarr {

typedef std::uint64_t zsize_t;

/**
 * Thrown when the array uses a feature that this reader does not implement.
 * Callers can catch this to fall back to another access method.
 */
class UnsupportedError : public std::runtime_error {
public:
    explicit UnsupportedError(const std::string& msg) : std::runtime_error(msg) {}
};

enum class DType { INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT32, FLOAT64, BOOL };

enum class Codec { BLOSC, ZLIB_OR_GZIP, ZSTD, LZ4, CRC32C };

struct ArrayMetadata {
    std::string path;
    int zarr_format = 2;
    std::vector<zsize_t> shape;
    std::vector<zsize_t> chunks;
    DType dtype = DType::FLOAT64;
    std::size_t itemsize = 8;
    bool big_endian = false;
    bool fortran_order = false;
    std::vector<Codec> codecs; // bytes-to-bytes codecs, in *encoding* order.
    std::vector<unsigned char> fill_bytes; // one element, native endianness.

    // Chunk key: prefix + c0 + sep + c1 + ...
    std::string key_prefix;
    std::string key_separator = ".";

    zsize_t chunk_elements() const {
        zsize_t n = 1;
        for (auto c : chunks) {
            n *= c;
        }
        return n;
    }

    zsize_t chunk_bytes() const {
        return chunk_elements() * itemsize;
    }

    std::string chunk_path(const std::vector<zsize_t>& coords) const {
        std::string out = path + "/" + key_prefix;
        for (std::size_t i = 0; i < coords.size(); ++i) {
            if (i) {
                out += key_separator;
            }
            out += std::to_string(coords[i]);
        }
        if (coords.empty()) {
            out += "0";
        }
        return out;
    }
};

namespace internal {

typedef std::shared_ptr<millijson::Base> JsonPtr;

inline bool file_exists(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f) {
        std::fclose(f);
        return true;
    }
    return false;
}

inline bool read_file(const std::string& path, std::vector<unsigned char>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    out.clear();
    unsigned char buf[65536];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        out.insert(out.end(), buf, buf + n);
    }
    bool err = std::ferror(f);
    std::fclose(f);
    if (err) {
        throw std::runtime_error("failed to read '" + path + "'");
    }
    return true;
}

inline const millijson::Object& as_object(const JsonPtr& x, const std::string& what) {
    if (!x || x->type() != millijson::OBJECT) {
        throw std::runtime_error("expected '" + what + "' to be a JSON object");
    }
    return static_cast<const millijson::Object&>(*x);
}

inline const JsonPtr* find(const millijson::Object& obj, const std::string& name) {
    const auto& m = obj.value();
    auto it = m.find(name);
    if (it == m.end()) {
        return nullptr;
    }
    return &(it->second);
}

inline const JsonPtr& get(const millijson::Object& obj, const std::string& name) {
    auto p = find(obj, name);
    if (!p) {
        throw std::runtime_error("missing '" + name + "' in Zarr metadata");
    }
    return *p;
}

inline std::string as_string(const JsonPtr& x, const std::string& what) {
    if (!x || x->type() != millijson::STRING) {
        throw std::runtime_error("expected '" + what + "' to be a string");
    }
    return static_cast<const millijson::String&>(*x).value();
}

inline std::vector<zsize_t> as_dims(const JsonPtr& x, const std::string& what) {
    if (!x || x->type() != millijson::ARRAY) {
        throw std::runtime_error("expected '" + what + "' to be an array");
    }
    std::vector<zsize_t> out;
    for (const auto& el : static_cast<const millijson::Array&>(*x).value()) {
        if (el->type() != millijson::NUMBER) {
            throw std::runtime_error("expected '" + what + "' to contain numbers");
        }
        double v = static_cast<const millijson::Number&>(*el).value();
        if (v < 0 || v != std::floor(v)) {
            throw std::runtime_error("expected '" + what + "' to contain non-negative integers");
        }
        out.push_back(static_cast<zsize_t>(v));
    }
    return out;
}

inline std::size_t dtype_size(DType t) {
    switch (t) {
        case DType::INT8: case DType::UINT8: case DType::BOOL: return 1;
        case DType::INT16: case DType::UINT16: return 2;
        case DType::INT32: case DType::UINT32: case DType::FLOAT32: return 4;
        default: return 8;
    }
}

inline bool is_float(DType t) {
    return t == DType::FLOAT32 || t == DType::FLOAT64;
}

// v2 typestr, e.g. "<f8", "|u1", ">i4".
inline void parse_v2_dtype(const std::string& s, ArrayMetadata& meta) {
    if (s.size() < 3) {
        throw UnsupportedError("unsupported Zarr dtype '" + s + "'");
    }
    char endian = s[0], kind = s[1];
    int size = std::atoi(s.c_str() + 2);
    meta.big_endian = (endian == '>');
    auto bad = [&]() { return UnsupportedError("unsupported Zarr dtype '" + s + "'"); };
    if (kind == 'f') {
        if (size == 4) meta.dtype = DType::FLOAT32;
        else if (size == 8) meta.dtype = DType::FLOAT64;
        else throw bad();
    } else if (kind == 'i') {
        if (size == 1) meta.dtype = DType::INT8;
        else if (size == 2) meta.dtype = DType::INT16;
        else if (size == 4) meta.dtype = DType::INT32;
        else if (size == 8) meta.dtype = DType::INT64;
        else throw bad();
    } else if (kind == 'u') {
        if (size == 1) meta.dtype = DType::UINT8;
        else if (size == 2) meta.dtype = DType::UINT16;
        else if (size == 4) meta.dtype = DType::UINT32;
        else if (size == 8) meta.dtype = DType::UINT64;
        else throw bad();
    } else if (kind == 'b' && size == 1) {
        meta.dtype = DType::BOOL;
    } else {
        throw bad();
    }
    meta.itemsize = dtype_size(meta.dtype);
}

inline void parse_v3_dtype(const JsonPtr& x, ArrayMetadata& meta) {
    if (!x || x->type() != millijson::STRING) {
        throw UnsupportedError("unsupported Zarr v3 data_type (not a simple string)");
    }
    const std::string& s = static_cast<const millijson::String&>(*x).value();
    if (s == "float32") meta.dtype = DType::FLOAT32;
    else if (s == "float64") meta.dtype = DType::FLOAT64;
    else if (s == "int8") meta.dtype = DType::INT8;
    else if (s == "int16") meta.dtype = DType::INT16;
    else if (s == "int32") meta.dtype = DType::INT32;
    else if (s == "int64") meta.dtype = DType::INT64;
    else if (s == "uint8") meta.dtype = DType::UINT8;
    else if (s == "uint16") meta.dtype = DType::UINT16;
    else if (s == "uint32") meta.dtype = DType::UINT32;
    else if (s == "uint64") meta.dtype = DType::UINT64;
    else if (s == "bool") meta.dtype = DType::BOOL;
    else throw UnsupportedError("unsupported Zarr v3 data_type '" + s + "'");
    meta.itemsize = dtype_size(meta.dtype);
}

template<typename T>
void store_fill(T v, ArrayMetadata& meta) {
    meta.fill_bytes.resize(sizeof(T));
    std::memcpy(meta.fill_bytes.data(), &v, sizeof(T));
}

inline void set_fill_from_double(double v, ArrayMetadata& meta) {
    switch (meta.dtype) {
        case DType::INT8: store_fill<std::int8_t>(static_cast<std::int8_t>(v), meta); break;
        case DType::UINT8: store_fill<std::uint8_t>(static_cast<std::uint8_t>(v), meta); break;
        case DType::BOOL: store_fill<std::uint8_t>(v != 0, meta); break;
        case DType::INT16: store_fill<std::int16_t>(static_cast<std::int16_t>(v), meta); break;
        case DType::UINT16: store_fill<std::uint16_t>(static_cast<std::uint16_t>(v), meta); break;
        case DType::INT32: store_fill<std::int32_t>(static_cast<std::int32_t>(v), meta); break;
        case DType::UINT32: store_fill<std::uint32_t>(static_cast<std::uint32_t>(v), meta); break;
        case DType::INT64: store_fill<std::int64_t>(static_cast<std::int64_t>(v), meta); break;
        case DType::UINT64: store_fill<std::uint64_t>(static_cast<std::uint64_t>(v), meta); break;
        case DType::FLOAT32: store_fill<float>(static_cast<float>(v), meta); break;
        case DType::FLOAT64: store_fill<double>(v, meta); break;
    }
}

inline void parse_fill(const JsonPtr* x, ArrayMetadata& meta) {
    double v = 0;
    if (x && *x) {
        switch ((*x)->type()) {
            case millijson::NUMBER:
                v = static_cast<const millijson::Number&>(**x).value();
                break;
            case millijson::BOOLEAN:
                v = static_cast<const millijson::Boolean&>(**x).value();
                break;
            case millijson::STRING: {
                const std::string& s = static_cast<const millijson::String&>(**x).value();
                if (s == "NaN") {
                    v = std::numeric_limits<double>::quiet_NaN();
                } else if (s == "Infinity") {
                    v = std::numeric_limits<double>::infinity();
                } else if (s == "-Infinity") {
                    v = -std::numeric_limits<double>::infinity();
                } else if (s.size() > 2 && s[0] == '0' && s[1] == 'x' && is_float(meta.dtype)) {
                    // v3 hex-encoded IEEE bits.
                    std::uint64_t bits = std::stoull(s.substr(2), nullptr, 16);
                    if (meta.dtype == DType::FLOAT32) {
                        std::uint32_t b32 = static_cast<std::uint32_t>(bits);
                        float f;
                        std::memcpy(&f, &b32, 4);
                        store_fill<float>(f, meta);
                    } else {
                        double d;
                        std::memcpy(&d, &bits, 8);
                        store_fill<double>(d, meta);
                    }
                    return;
                } else {
                    throw UnsupportedError("unsupported fill_value '" + s + "'");
                }
                break;
            }
            default:
                break; // null or other: treat as zero.
        }
    }
    set_fill_from_double(v, meta);
}

inline Codec parse_compressor_id(const std::string& id) {
    if (id == "blosc") {
#ifdef TATAMI_ZARR_HAS_BLOSC
        return Codec::BLOSC;
#else
        throw UnsupportedError("beachmat.zarr was built without blosc support");
#endif
    } else if (id == "zlib" || id == "gzip") {
        return Codec::ZLIB_OR_GZIP;
    } else if (id == "zstd") {
#ifdef TATAMI_ZARR_HAS_ZSTD
        return Codec::ZSTD;
#else
        throw UnsupportedError("beachmat.zarr was built without zstd support");
#endif
    } else if (id == "lz4") {
#ifdef TATAMI_ZARR_HAS_LZ4
        return Codec::LZ4;
#else
        throw UnsupportedError("beachmat.zarr was built without lz4 support");
#endif
    } else if (id == "crc32c") {
        return Codec::CRC32C;
    }
    throw UnsupportedError("unsupported Zarr codec '" + id + "'");
}

inline void parse_v2(const std::string& path, ArrayMetadata& meta) {
    auto root = millijson::parse_file((path + "/.zarray").c_str(), millijson::ParseOptions());
    const auto& obj = as_object(root, ".zarray");
    meta.zarr_format = 2;
    meta.shape = as_dims(get(obj, "shape"), "shape");
    meta.chunks = as_dims(get(obj, "chunks"), "chunks");

    const auto& dt = get(obj, "dtype");
    if (dt->type() != millijson::STRING) {
        throw UnsupportedError("structured Zarr dtypes are not supported");
    }
    parse_v2_dtype(static_cast<const millijson::String&>(*dt).value(), meta);

    if (auto o = find(obj, "order")) {
        meta.fortran_order = (as_string(*o, "order") == "F");
    }

    if (auto f = find(obj, "filters")) {
        if (*f && (*f)->type() == millijson::ARRAY && !static_cast<const millijson::Array&>(**f).value().empty()) {
            throw UnsupportedError("Zarr v2 filters are not supported");
        }
    }

    if (auto c = find(obj, "compressor")) {
        if (*c && (*c)->type() == millijson::OBJECT) {
            const auto& cobj = static_cast<const millijson::Object&>(**c);
            meta.codecs.push_back(parse_compressor_id(as_string(get(cobj, "id"), "compressor.id")));
        }
    }

    if (auto s = find(obj, "dimension_separator")) {
        meta.key_separator = as_string(*s, "dimension_separator");
    }
    meta.key_prefix = "";

    parse_fill(find(obj, "fill_value"), meta);
}

inline void parse_v3(const std::string& path, ArrayMetadata& meta) {
    auto root = millijson::parse_file((path + "/zarr.json").c_str(), millijson::ParseOptions());
    const auto& obj = as_object(root, "zarr.json");
    meta.zarr_format = 3;

    if (as_string(get(obj, "node_type"), "node_type") != "array") {
        throw std::runtime_error("'" + path + "' is not a Zarr array");
    }
    meta.shape = as_dims(get(obj, "shape"), "shape");
    parse_v3_dtype(get(obj, "data_type"), meta);

    const auto& grid = as_object(get(obj, "chunk_grid"), "chunk_grid");
    if (as_string(get(grid, "name"), "chunk_grid.name") != "regular") {
        throw UnsupportedError("only regular Zarr chunk grids are supported");
    }
    meta.chunks = as_dims(get(as_object(get(grid, "configuration"), "chunk_grid.configuration"), "chunk_shape"), "chunk_shape");

    // Chunk key encoding.
    meta.key_prefix = "c/";
    meta.key_separator = "/";
    if (auto cke = find(obj, "chunk_key_encoding")) {
        const auto& ckeo = as_object(*cke, "chunk_key_encoding");
        std::string name = as_string(get(ckeo, "name"), "chunk_key_encoding.name");
        std::string sep;
        if (auto conf = find(ckeo, "configuration")) {
            if (auto s = find(as_object(*conf, "chunk_key_encoding.configuration"), "separator")) {
                sep = as_string(*s, "separator");
            }
        }
        if (name == "default") {
            meta.key_separator = sep.empty() ? "/" : sep;
            meta.key_prefix = "c" + meta.key_separator;
        } else if (name == "v2") {
            meta.key_separator = sep.empty() ? "." : sep;
            meta.key_prefix = "";
        } else {
            throw UnsupportedError("unsupported chunk_key_encoding '" + name + "'");
        }
    }

    // Codec pipeline: array->array, array->bytes, bytes->bytes.
    const auto& codecs = get(obj, "codecs");
    if (codecs->type() != millijson::ARRAY) {
        throw std::runtime_error("expected 'codecs' to be an array");
    }
    bool seen_bytes = false;
    for (const auto& c : static_cast<const millijson::Array&>(*codecs).value()) {
        const auto& co = as_object(c, "codec");
        std::string name = as_string(get(co, "name"), "codec.name");
        const millijson::Object* conf = nullptr;
        if (auto cf = find(co, "configuration")) {
            if (*cf && (*cf)->type() == millijson::OBJECT) {
                conf = static_cast<const millijson::Object*>(cf->get());
            }
        }

        if (name == "transpose") {
            if (seen_bytes) {
                throw UnsupportedError("transpose codec after bytes codec");
            }
            if (!conf) {
                throw std::runtime_error("transpose codec requires a configuration");
            }
            auto order = as_dims(get(*conf, "order"), "transpose.order");
            bool identity = true, reversed = true;
            for (std::size_t i = 0; i < order.size(); ++i) {
                identity = identity && order[i] == i;
                reversed = reversed && order[i] == order.size() - 1 - i;
            }
            if (identity) {
                // no-op
            } else if (reversed) {
                meta.fortran_order = !meta.fortran_order;
            } else {
                throw UnsupportedError("only identity/reversing transpose codecs are supported");
            }
        } else if (name == "bytes") {
            seen_bytes = true;
            if (conf) {
                if (auto e = find(*conf, "endian")) {
                    meta.big_endian = (as_string(*e, "endian") == "big");
                }
            }
        } else if (name == "sharding_indexed") {
            throw UnsupportedError("sharded Zarr arrays are not supported");
        } else {
            if (!seen_bytes) {
                throw UnsupportedError("unsupported array-to-array codec '" + name + "'");
            }
            meta.codecs.push_back(parse_compressor_id(name));
        }
    }
    if (!seen_bytes) {
        throw UnsupportedError("Zarr v3 array is missing the 'bytes' codec");
    }

    parse_fill(find(obj, "fill_value"), meta);
}

inline void byteswap(unsigned char* data, std::size_t n, std::size_t itemsize) {
    if (itemsize == 1) {
        return;
    }
    for (std::size_t i = 0; i < n; ++i) {
        std::reverse(data + i * itemsize, data + (i + 1) * itemsize);
    }
}

inline bool host_is_big_endian() {
    const std::uint16_t x = 1;
    unsigned char c;
    std::memcpy(&c, &x, 1);
    return c == 0;
}

inline void inflate_any(const std::vector<unsigned char>& in, std::vector<unsigned char>& out, std::size_t expected) {
    out.resize(expected);
    z_stream strm;
    std::memset(&strm, 0, sizeof(strm));
    if (inflateInit2(&strm, 15 + 32) != Z_OK) { // auto-detect zlib or gzip header.
        throw std::runtime_error("failed to initialize zlib");
    }
    strm.next_in = const_cast<unsigned char*>(in.data());
    strm.avail_in = static_cast<uInt>(in.size());
    strm.next_out = out.data();
    strm.avail_out = static_cast<uInt>(out.size());
    int ret = inflate(&strm, Z_FINISH);
    std::size_t produced = out.size() - strm.avail_out;
    inflateEnd(&strm);
    if (ret != Z_STREAM_END && !(ret == Z_BUF_ERROR && produced == expected)) {
        throw std::runtime_error("zlib/gzip decompression failed");
    }
    out.resize(produced);
}

inline void decode_one(Codec codec, const std::vector<unsigned char>& in, std::vector<unsigned char>& out, std::size_t expected) {
    switch (codec) {
        case Codec::ZLIB_OR_GZIP:
            inflate_any(in, out, expected);
            return;
        case Codec::CRC32C:
            if (in.size() < 4) {
                throw std::runtime_error("crc32c-encoded chunk is too short");
            }
            out.assign(in.begin(), in.end() - 4);
            return;
#ifdef TATAMI_ZARR_HAS_BLOSC
        case Codec::BLOSC: {
            std::size_t nbytes = 0, cbytes = 0, blocksize = 0;
            blosc_cbuffer_sizes(in.data(), &nbytes, &cbytes, &blocksize);
            out.resize(nbytes);
            int n = blosc_decompress_ctx(in.data(), out.data(), out.size(), 1);
            if (n < 0) {
                throw std::runtime_error("blosc decompression failed");
            }
            out.resize(n);
            return;
        }
#endif
#ifdef TATAMI_ZARR_HAS_ZSTD
        case Codec::ZSTD: {
            unsigned long long sz = ZSTD_getFrameContentSize(in.data(), in.size());
            std::size_t cap = (sz == ZSTD_CONTENTSIZE_UNKNOWN || sz == ZSTD_CONTENTSIZE_ERROR) ? expected : static_cast<std::size_t>(sz);
            out.resize(cap);
            std::size_t n = ZSTD_decompress(out.data(), out.size(), in.data(), in.size());
            if (ZSTD_isError(n)) {
                throw std::runtime_error(std::string("zstd decompression failed: ") + ZSTD_getErrorName(n));
            }
            out.resize(n);
            return;
        }
#endif
#ifdef TATAMI_ZARR_HAS_LZ4
        case Codec::LZ4: {
            // numcodecs framing: 4-byte little-endian uncompressed size, then a raw LZ4 block.
            if (in.size() < 4) {
                throw std::runtime_error("lz4-encoded chunk is too short");
            }
            std::uint32_t sz = static_cast<std::uint32_t>(in[0]) | (static_cast<std::uint32_t>(in[1]) << 8) |
                (static_cast<std::uint32_t>(in[2]) << 16) | (static_cast<std::uint32_t>(in[3]) << 24);
            out.resize(sz);
            int n = LZ4_decompress_safe(reinterpret_cast<const char*>(in.data()) + 4, reinterpret_cast<char*>(out.data()), static_cast<int>(in.size() - 4), static_cast<int>(sz));
            if (n < 0) {
                throw std::runtime_error("lz4 decompression failed");
            }
            out.resize(n);
            return;
        }
#endif
        default:
            throw UnsupportedError("codec not available in this build");
    }
}

} // namespace internal

/**
 * Parse metadata for the Zarr array stored in the directory `path`.
 * Both v3 (`zarr.json`) and v2 (`.zarray`) are recognized.
 */
inline std::shared_ptr<const ArrayMetadata> read_metadata(const std::string& path) {
    auto meta = std::make_shared<ArrayMetadata>();
    meta->path = path;
    while (meta->path.size() > 1 && meta->path.back() == '/') {
        meta->path.pop_back();
    }
    if (internal::file_exists(meta->path + "/zarr.json")) {
        internal::parse_v3(meta->path, *meta);
    } else if (internal::file_exists(meta->path + "/.zarray")) {
        internal::parse_v2(meta->path, *meta);
    } else {
        throw std::runtime_error("no Zarr array metadata found at '" + path + "'");
    }
    if (meta->chunks.size() != meta->shape.size()) {
        throw std::runtime_error("Zarr 'chunks' and 'shape' have different lengths at '" + path + "'");
    }
    for (auto c : meta->chunks) {
        if (c == 0) {
            throw std::runtime_error("Zarr chunk dimensions must be positive at '" + path + "'");
        }
    }
    return meta;
}

/**
 * Decodes whole chunks into raw native-endian bytes.
 * Not thread-safe; create one per thread/extractor.
 */
class ChunkReader {
public:
    explicit ChunkReader(std::shared_ptr<const ArrayMetadata> meta) : my_meta(std::move(meta)) {}

    const ArrayMetadata& metadata() const { return *my_meta; }

    /**
     * @param coords Chunk grid coordinates.
     * @param[out] out Decoded chunk of `chunk_bytes()` bytes, in the array's chunk layout (C or F).
     */
    void read(const std::vector<zsize_t>& coords, std::vector<unsigned char>& out) {
        const auto& meta = *my_meta;
        const std::size_t expected = meta.chunk_bytes();
        const std::string cpath = meta.chunk_path(coords);

        if (!internal::read_file(cpath, my_buffer_a)) {
            // Missing chunk: fill.
            out.resize(expected);
            const std::size_t isz = meta.itemsize;
            for (std::size_t i = 0; i < expected; i += isz) {
                std::memcpy(out.data() + i, meta.fill_bytes.data(), isz);
            }
            return;
        }

        // Decode bytes-to-bytes codecs in reverse order.
        std::vector<unsigned char>* cur = &my_buffer_a;
        std::vector<unsigned char>* nxt = &my_buffer_b;
        for (auto it = meta.codecs.rbegin(); it != meta.codecs.rend(); ++it) {
            internal::decode_one(*it, *cur, *nxt, expected);
            std::swap(cur, nxt);
        }

        if (cur->size() != expected) {
            throw std::runtime_error("decoded chunk '" + cpath + "' has " + std::to_string(cur->size()) + " bytes, expected " + std::to_string(expected));
        }
        out.swap(*cur);

        if (meta.big_endian != internal::host_is_big_endian()) {
            internal::byteswap(out.data(), meta.chunk_elements(), meta.itemsize);
        }
    }

private:
    std::shared_ptr<const ArrayMetadata> my_meta;
    std::vector<unsigned char> my_buffer_a, my_buffer_b;
};

/**
 * Convert `n` elements starting at element `offset` of a decoded chunk into `T`.
 */
template<typename T>
void convert(const unsigned char* raw, DType dtype, std::size_t offset, std::size_t n, T* out) {
    auto loop = [&](auto dummy) {
        typedef decltype(dummy) S;
        const unsigned char* src = raw + offset * sizeof(S);
        for (std::size_t i = 0; i < n; ++i) {
            S v;
            std::memcpy(&v, src + i * sizeof(S), sizeof(S));
            out[i] = static_cast<T>(v);
        }
    };
    switch (dtype) {
        case DType::INT8: loop(std::int8_t()); break;
        case DType::UINT8: loop(std::uint8_t()); break;
        case DType::BOOL: loop(std::uint8_t()); break;
        case DType::INT16: loop(std::int16_t()); break;
        case DType::UINT16: loop(std::uint16_t()); break;
        case DType::INT32: loop(std::int32_t()); break;
        case DType::UINT32: loop(std::uint32_t()); break;
        case DType::INT64: loop(std::int64_t()); break;
        case DType::UINT64: loop(std::uint64_t()); break;
        case DType::FLOAT32: loop(float()); break;
        case DType::FLOAT64: loop(double()); break;
    }
}

/**
 * Reads arbitrary element ranges from a 1-dimensional Zarr array,
 * keeping a small LRU cache of decoded chunks so that consecutive
 * reads that share a chunk only decode it once.
 * Not thread-safe; create one per thread/extractor.
 */
class Array1DReader {
public:
    Array1DReader(std::shared_ptr<const ArrayMetadata> meta, std::size_t cache_chunks = 2) :
        my_reader(meta), my_meta(std::move(meta)), my_cache(std::max<std::size_t>(cache_chunks, 1))
    {
        if (my_meta->shape.size() != 1) {
            throw std::runtime_error("expected a 1-dimensional Zarr array at '" + my_meta->path + "'");
        }
    }

    zsize_t length() const { return my_meta->shape[0]; }

    const ArrayMetadata& metadata() const { return *my_meta; }

    template<typename T>
    void read(zsize_t start, zsize_t len, T* out) {
        const zsize_t csize = my_meta->chunks[0];
        zsize_t end = start + len;
        while (start < end) {
            zsize_t cid = start / csize;
            zsize_t cstart = cid * csize;
            zsize_t local = start - cstart;
            zsize_t take = std::min(end, cstart + csize) - start;
            const auto& chunk = fetch(cid);
            convert(chunk.data(), my_meta->dtype, local, take, out);
            out += take;
            start += take;
        }
    }

private:
    struct Entry {
        zsize_t id = static_cast<zsize_t>(-1);
        std::uint64_t used = 0;
        std::vector<unsigned char> data;
    };

    const std::vector<unsigned char>& fetch(zsize_t cid) {
        ++my_clock;
        Entry* victim = &my_cache[0];
        for (auto& e : my_cache) {
            if (e.id == cid) {
                e.used = my_clock;
                return e.data;
            }
            if (e.used < victim->used) {
                victim = &e;
            }
        }
        my_coords.assign(1, cid);
        my_reader.read(my_coords, victim->data);
        victim->id = cid;
        victim->used = my_clock;
        return victim->data;
    }

    ChunkReader my_reader;
    std::shared_ptr<const ArrayMetadata> my_meta;
    std::vector<Entry> my_cache;
    std::vector<zsize_t> my_coords;
    std::uint64_t my_clock = 0;
};

}

#endif
