#include <simt_tf/matrix.h>

#include <ostream>

namespace simt_tf {

std::ostream& operator<<(std::ostream &os, const Matrix &m) {
    return os << '{' << m[0] << ", " << m[1] << ", " << m[2] << '}';
}

} // namespace simt_tf
