#include <simt_tf/matrix.h>

#include <ostream>

namespace simt_tf {

/**
 *  Serializes a matrix to a std::ostream.
 *
 *  Elements are output in row-major order on a single line, like:
 *  {{a00, a01, a02}, {a10, a11, a12}, {a20, a21, a22}}.
 */
std::ostream& operator<<(std::ostream &os, const Matrix &m) {
    return os << '{' << m[0] << ", " << m[1] << ", " << m[2] << '}';
}

} // namespace simt_tf
