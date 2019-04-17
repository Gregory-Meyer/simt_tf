#include <simt_tf/vector.h>

#include <ostream>

namespace simt_tf {

/**
 *  Serializes a Vector to an ostream.
 *
 *  Elements are serialized on one line, in the format:
 *  {x, y, z}
 */
std::ostream& operator<<(std::ostream &os, const Vector &v) {
    return os << '{' << v.x() << ", " << v.y() << ", " << v.z() << '}';
}

} // namespace simt_tf
