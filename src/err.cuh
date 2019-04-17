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
