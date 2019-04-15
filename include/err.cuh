#ifndef ERR_CUH
#define ERR_CUH

#include <system_error>
#include <type_traits>

namespace std {

template <>
struct is_error_code_enum<cudaError_t> : std::true_type { };

} // namespace std

const std::error_category& cuda_category() noexcept;

std::error_code make_error_code(cudaError_t error) noexcept;

#endif
