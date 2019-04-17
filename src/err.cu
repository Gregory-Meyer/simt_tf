// Copyright (c) 2019 Gregory Meyer
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice (including
// the next paragraph) shall be included in all copies or substantial
// portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include "err.cuh"

#include <cstring>
#include <string>

class CudaCategory : public std::error_category {
public:
    CudaCategory() noexcept = default;

    virtual ~CudaCategory() = default;

    const char* name() const noexcept override {
        return "CudaCategory";
    }

    std::string message(int condition) const override {
        const auto err = static_cast<cudaError_t>(condition);

        const char *const name = cudaGetErrorName(err);
        const char *const desc = cudaGetErrorString(err);

        const std::size_t name_len = std::strlen(name);
        const std::size_t desc_len = std::strlen(desc);

        std::string msg;
        msg.reserve(name_len + desc_len + 2);

        msg.append(name, name_len);
        msg.append(": ", 2);
        msg.append(desc, desc_len);

        return msg;
    }
};

const std::error_category& cuda_category() noexcept {
    static const CudaCategory category;

    return category;
}

std::error_code make_error_code(cudaError_t error) noexcept {
    return {static_cast<int>(error), cuda_category()};
}

namespace sl {

class SlCategory : public std::error_category {
public:
    SlCategory() noexcept = default;

    virtual ~SlCategory() = default;

    const char* name() const noexcept override {
        return "sl::SlCategory";
    }

    std::string message(int condition) const override {
        const String msg = toString(static_cast<ERROR_CODE>(condition));

        return {msg.c_str()};
    }
};

const std::error_category& sl_category() noexcept {
    static const SlCategory category;

    return category;
}

std::error_code make_error_code(ERROR_CODE error) noexcept {
    return {static_cast<int>(error), sl_category()};
}

} // namespace sl
