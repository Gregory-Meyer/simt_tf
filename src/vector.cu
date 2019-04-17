#include <simt_tf/vector.h>

#include <ostream>

namespace simt_tf {

std::ostream& operator<<(std::ostream &os, const Vector &v) {
    return os << '{' << v.x() << ", " << v.y() << ", " << v.z() << '}';
}

} // namespace simt_tf
