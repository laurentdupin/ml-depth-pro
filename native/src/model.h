#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

namespace depth_pro_native {

struct TensorView {
    const std::uint16_t* data = nullptr;
    std::array<std::uint64_t, 4> dimensions{};
    std::uint32_t rank = 0;
    std::uint64_t elements = 0;
};

struct Derivation {
    std::array<std::uint8_t, 32> canonical_sha256{};
    std::string converter;
    std::uint32_t format_version = 0;
};

float half_to_float(std::uint16_t value);

class ModelFile {
public:
    explicit ModelFile(const std::string& path_utf8);
    ModelFile(const ModelFile&) = delete;
    ModelFile& operator=(const ModelFile&) = delete;
    ~ModelFile();

    const TensorView& tensor(std::string_view name) const;
    bool contains(std::string_view name) const;
    std::size_t tensor_count() const { return tensors_.size(); }
    const Derivation& derivation() const { return derivation_; }

private:
    void close() noexcept;

#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int file_descriptor_ = -1;
#endif
    const std::byte* view_ = nullptr;
    std::uint64_t size_ = 0;
    std::unordered_map<std::string, TensorView> tensors_;
    Derivation derivation_;
};

}  // namespace depth_pro_native

