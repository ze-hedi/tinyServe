#pragma once

#include "json.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct MMapLoadOptions {
    bool prefault = false;
    bool huge_pages = false;
};

class ReadOnlyMapping {
public:
    ReadOnlyMapping() = default;

    explicit ReadOnlyMapping(const std::filesystem::path& path,
                             const MMapLoadOptions& options = {}) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ == -1) {
            throw std::runtime_error("mmap loader: cannot open '" + path.string() + "'");
        }

        struct stat info {};
        if (::fstat(fd_, &info) == -1 || info.st_size < 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("mmap loader: cannot stat '" + path.string() + "'");
        }
        size_ = static_cast<std::size_t>(info.st_size);
        if (size_ == 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("mmap loader: file is empty: '" + path.string() + "'");
        }

        int flags = MAP_PRIVATE;
        if (options.prefault) {
#ifdef MAP_POPULATE
            flags |= MAP_POPULATE;
#else
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("mmap loader: prefault is unsupported on this platform");
#endif
        }
        data_ = ::mmap(nullptr, size_, PROT_READ, flags, fd_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("mmap loader: mmap failed for '" + path.string() + "'");
        }
        if (options.huge_pages) {
#ifdef MADV_HUGEPAGE
            if (::madvise(data_, size_, MADV_HUGEPAGE) != 0) {
                ::munmap(data_, size_);
                data_ = nullptr;
                ::close(fd_);
                fd_ = -1;
                throw std::runtime_error(
                    "mmap loader: MADV_HUGEPAGE failed for '" + path.string() + "'");
            }
#else
            ::munmap(data_, size_);
            data_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("mmap loader: huge pages are unsupported on this platform");
#endif
        }
    }

    ~ReadOnlyMapping() {
        if (data_) ::munmap(data_, size_);
        if (fd_ != -1) ::close(fd_);
    }

    ReadOnlyMapping(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping& operator=(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping(ReadOnlyMapping&& other) noexcept
        : fd_{other.fd_}, data_{other.data_}, size_{other.size_} {
        other.fd_ = -1;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    ReadOnlyMapping& operator=(ReadOnlyMapping&& other) noexcept {
        if (this != &other) {
            if (data_) ::munmap(data_, size_);
            if (fd_ != -1) ::close(fd_);
            fd_ = other.fd_;
            data_ = other.data_;
            size_ = other.size_;
            other.fd_ = -1;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    const void* data() const { return data_; }
    std::size_t size() const { return size_; }

private:
    int fd_{-1};
    void* data_{nullptr};
    std::size_t size_{0};
};

// Read-only weight matrix backed by mmap. The mapping must stay alive for
// the lifetime of this object — data is never copied.
class MMapWeight {
public:
    MMapWeight() = default;

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    const float* data() const { return data_; }

    // Parse manifest entry, mmap the .bin file, and reinterpret as float*.
    void load(const std::filesystem::path& directory,
              const nlohmann::json& manifest,
              const std::string& name,
              const MMapLoadOptions& options = {}) {
        const nlohmann::json* entry = nullptr;
        for (const auto& tensor : manifest.at("tensors")) {
            if (tensor.at("name").get<std::string>() == name) {
                entry = &tensor;
                break;
            }
        }
        if (!entry) {
            throw std::runtime_error("mmap loader: manifest is missing tensor '" + name + "'");
        }
        if (entry->at("dtype") != "float32" || entry->at("layout") != "row-major" ||
            entry->at("byte_order") != "little-endian") {
            throw std::runtime_error("mmap loader: unsupported format for tensor '" + name + "'");
        }
        if constexpr (std::endian::native != std::endian::little) {
            throw std::runtime_error(
                "mmap loader: little-endian float32 requires a little-endian host");
        }

        const auto& shape = entry->at("shape");
        if (!shape.is_array() || shape.size() != 2) {
            throw std::runtime_error("mmap loader: expected a 2D tensor for '" + name + "'");
        }
        rows_ = shape[0].get<std::size_t>();
        cols_ = shape[1].get<std::size_t>();
        if (cols_ != 0 && rows_ > std::numeric_limits<std::size_t>::max() / cols_) {
            throw std::runtime_error("mmap loader: shape overflows for tensor '" + name + "'");
        }
        const std::size_t elements = rows_ * cols_;
        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::runtime_error("mmap loader: tensor is too large: '" + name + "'");
        }
        const std::size_t byte_count = elements * sizeof(float);
        if (entry->at("size_bytes").get<std::size_t>() != byte_count) {
            throw std::runtime_error("mmap loader: manifest byte size mismatch for '" + name + "'");
        }

        const auto filename = entry->at("file").get<std::string>();
        const std::filesystem::path relative_path(filename);
        if (relative_path.is_absolute() || relative_path.has_parent_path()) {
            throw std::runtime_error("mmap loader: tensor filename must be a simple file name");
        }

        mapping_ = ReadOnlyMapping(directory / relative_path, options);
        if (mapping_.size() != byte_count) {
            throw std::runtime_error("mmap loader: file size mismatch for tensor '" + name + "'");
        }

        data_ = reinterpret_cast<const float*>(mapping_.data());
    }

private:
    ReadOnlyMapping mapping_{};
    const float* data_{nullptr};
    std::size_t rows_{0};
    std::size_t cols_{0};
};

struct MMapLlamaBlockWeights {
    MMapWeight input_layernorm;
    MMapWeight q_proj;
    MMapWeight k_proj;
    MMapWeight v_proj;
    MMapWeight o_proj;
    MMapWeight post_attention_layernorm;
    MMapWeight gate_proj;
    MMapWeight up_proj;
    MMapWeight down_proj;
};

inline MMapLlamaBlockWeights load_mapped_llama_block(
        const std::filesystem::path& directory,
        const MMapLoadOptions& options = {}) {
    const auto manifest_path = directory / "shapes.json";
    std::ifstream manifest_file(manifest_path);
    if (!manifest_file) {
        throw std::runtime_error("mmap loader: cannot open '" + manifest_path.string() + "'");
    }

    nlohmann::json manifest;
    try {
        manifest = nlohmann::json::parse(manifest_file);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("mmap loader: invalid shapes.json: ") + error.what());
    }
    if (manifest.at("format") != "tinyserve-float32-matrices" ||
        manifest.at("format_version") != 1) {
        throw std::runtime_error("mmap loader: unsupported block manifest format");
    }

    MMapLlamaBlockWeights weights;
    weights.input_layernorm.load(directory, manifest, "input_layernorm", options);
    weights.q_proj.load(directory, manifest, "q_proj", options);
    weights.k_proj.load(directory, manifest, "k_proj", options);
    weights.v_proj.load(directory, manifest, "v_proj", options);
    weights.o_proj.load(directory, manifest, "o_proj", options);
    weights.post_attention_layernorm.load(directory, manifest, "post_attention_layernorm", options);
    weights.gate_proj.load(directory, manifest, "gate_proj", options);
    weights.up_proj.load(directory, manifest, "up_proj", options);
    weights.down_proj.load(directory, manifest, "down_proj", options);
    return weights;
}
