#include "matrix.cuh"

#include <ostream>

std::ostream& operator<<(std::ostream &os, const Matrix &m) {
    return os << '{' << m[0] << ", " << m[1] << ", " << m[2] << '}';
}
