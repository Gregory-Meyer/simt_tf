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

#ifndef SIMT_TF_IMPL_ERR_CUH
#define SIMT_TF_IMPL_ERR_CUH

#include <system_error>
#include <type_traits>

#include <sl/Core.hpp>

namespace std {

template <>
struct is_error_code_enum<cudaError_t> : std::true_type { };

template <>
struct is_error_code_enum<sl::ERROR_CODE> : std::true_type { };

} // namespace std

const std::error_category& cuda_category() noexcept;

std::error_code make_error_code(cudaError_t error) noexcept;

namespace sl {

std::error_code make_error_code(ERROR_CODE error) noexcept;

const std::error_category& sl_category() noexcept;

}

#endif
