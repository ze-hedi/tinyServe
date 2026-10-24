#pragma once

#include "json.hpp"
#include "mat.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// First-layer weights, converted from the checkpoint's BF16 storage to float.
// Linear weights are transposed from Hugging Face's [out_features, in_features]
// layout into the [in_features, out_features] layout consumed by LlamaBlock.
struct FirstLlamaBlockWeights {
    Mat<float> input_layernorm;
    Mat<float> q_proj;
    Mat<float> k_proj;
    Mat<float> v_proj;
    Mat<float> o_proj;
    Mat<float> post_attention_layernorm;
    Mat<float> gate_proj;
    Mat<float> up_proj;
    Mat<float> down_proj;
};

namespace safetensors_detail {

struct TensorInfo {
    std::vector<std::size_t> shape;
    std::size_t start;
    std::size_t end;
};

inline std::uint64_t read_u64_le(std::ifstream& input) {
    unsigned char bytes[8];
    input.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    if (!input) {
        throw std::runtime_error("safetensors: unable to read header length");
    }
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < sizeof(bytes); ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

inline std::size_t element_count(const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    for (std::size_t dim : shape) {
        if (dim != 0 && count > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("safetensors: tensor shape overflows address space");
        }
        count *= dim;
    }
    return count;
}

inline TensorInfo tensor_info(const nlohmann::json& header, const std::string& name) {
    if (!header.contains(name)) {
        throw std::runtime_error("safetensors: missing tensor '" + name + "'");
    }
    const auto& entry = header.at(name);
    if (entry.at("dtype") != "BF16") {
        throw std::runtime_error("safetensors: tensor '" + name + "' is not BF16");
    }

    TensorInfo info;
    for (const auto& dim : entry.at("shape")) {
        info.shape.push_back(dim.get<std::size_t>());
    }
    const auto& offsets = entry.at("data_offsets");
    if (!offsets.is_array() || offsets.size() != 2) {
        throw std::runtime_error("safetensors: invalid data_offsets for '" + name + "'");
    }
    info.start = offsets[0].get<std::size_t>();
    info.end = offsets[1].get<std::size_t>();

    const std::size_t count = element_count(info.shape);
    if (count > std::numeric_limits<std::size_t>::max() / 2 ||
        info.end < info.start || info.end - info.start != count * 2) {
        throw std::runtime_error("safetensors: invalid byte range for '" + name + "'");
    }
    return info;
}

inline std::vector<float> read_bf16(std::ifstream& input, std::streamoff data_start,
                                    const std::string& name, const TensorInfo& info) {
    const std::size_t count = element_count(info.shape);
    if (info.start > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max()) ||
        data_start > std::numeric_limits<std::streamoff>::max() -
                         static_cast<std::streamoff>(info.start)) {
        throw std::runtime_error("safetensors: tensor offset is too large for '" + name + "'");
    }
    input.clear();
    input.seekg(data_start + static_cast<std::streamoff>(info.start));
    std::vector<unsigned char> bytes(count * 2);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        throw std::runtime_error("safetensors: truncated data for tensor '" + name + "'");
    }

    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        // Safetensors stores little-endian BF16. Expanding its bits to the high
        // half of an IEEE float preserves the BF16 value exactly.
        const std::uint32_t bits =
            (static_cast<std::uint32_t>(bytes[2 * i + 1]) << 8) | bytes[2 * i];
        values[i] = std::bit_cast<float>(bits << 16);
    }
    return values;
}

inline Mat<float> read_vector(std::ifstream& input, std::streamoff data_start,
                              const nlohmann::json& header, const std::string& name) {
    const auto info = tensor_info(header, name);
    if (info.shape.size() != 1) {
        throw std::runtime_error("safetensors: expected a vector for '" + name + "'");
    }
    const auto values = read_bf16(input, data_start, name, info);
    return Mat<float>(1, values.size(), values);
}

inline Mat<float> read_linear(std::ifstream& input, std::streamoff data_start,
                              const nlohmann::json& header, const std::string& name) {
    const auto info = tensor_info(header, name);
    if (info.shape.size() != 2) {
        throw std::runtime_error("safetensors: expected a matrix for '" + name + "'");
    }
    const std::size_t out_features = info.shape[0];
    const std::size_t in_features = info.shape[1];
    auto values = read_bf16(input, data_start, name, info);

    // Checkpoint storage is [out, in], while this project's Mat convention for
    // projection weights is [in, out].
    std::vector<float> transposed(values.size());
    for (std::size_t out = 0; out < out_features; ++out) {
        for (std::size_t in = 0; in < in_features; ++in) {
            transposed[in * out_features + out] = values[out * in_features + in];
        }
    }
    return Mat<float>(in_features, out_features, std::move(transposed));
}

} // namespace safetensors_detail

// Load model.layers.0 from a Hugging Face Llama-family BF16 .safetensors file.
// For sharded checkpoints, pass the shard that contains all first-layer tensors.
inline FirstLlamaBlockWeights load_first_llama_block(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("safetensors: cannot open file '" + path + "'");
    }

    const std::uint64_t header_size = safetensors_detail::read_u64_le(input);
    if (header_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        header_size > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("safetensors: header is too large");
    }
    std::string header_text(static_cast<std::size_t>(header_size), '\0');
    input.read(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    if (!input) {
        throw std::runtime_error("safetensors: truncated JSON header");
    }

    nlohmann::json header;
    try {
        header = nlohmann::json::parse(header_text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("safetensors: invalid JSON header: ") + error.what());
    }
    const std::streamoff data_start = static_cast<std::streamoff>(8 + header_size);
    constexpr const char* prefix = "model.layers.0.";
    const auto tensor = [&](const std::string& suffix) {
        return std::string(prefix) + suffix;
    };

    return FirstLlamaBlockWeights{
        safetensors_detail::read_vector(input, data_start, header,
                                        tensor("input_layernorm.weight")),
        safetensors_detail::read_linear(input, data_start, header,
                                       tensor("self_attn.q_proj.weight")),
        safetensors_detail::read_linear(input, data_start, header,
                                       tensor("self_attn.k_proj.weight")),
        safetensors_detail::read_linear(input, data_start, header,
                                       tensor("self_attn.v_proj.weight")),
        safetensors_detail::read_linear(input, data_start, header,
                                       tensor("self_attn.o_proj.weight")),
        safetensors_detail::read_vector(input, data_start, header,
                                        tensor("post_attention_layernorm.weight")),
        safetensors_detail::read_linear(input, data_start, header,
                                       tensor("mlp.gate_proj.weight")),
        safetensors_detail::read_linear(input, data_start, header,
                                       tensor("mlp.up_proj.weight")),
        safetensors_detail::read_linear(input, data_start, header,
                                       tensor("mlp.down_proj.weight"))};
}
