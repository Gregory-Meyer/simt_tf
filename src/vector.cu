#include "vector.cuh"

#include <ostream>

std::ostream& operator<<(std::ostream &os, const Vector &v) {
    return os << '{' << v.x() << ", " << v.y() << ", " << v.z() << '}';
}
