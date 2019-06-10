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

#ifndef SIMT_TF_MATRIX_SPAN_H
#define SIMT_TF_MATRIX_SPAN_H

#include <simt_tf/macro.h>

#include <cassert>
#include <cstddef>

namespace simt_tf {

template <typename T> class RowSpan;

/**
 *  A dynamically-sized view into a Matrix.
 */
template <typename T> class MatrixSpan {
public:
  SIMT_TF_HOST_DEVICE constexpr MatrixSpan(T *data, std::size_t num_rows,
                                           std::size_t num_cols) noexcept
      : MatrixSpan(data, num_rows, num_cols, num_cols) {}

  SIMT_TF_HOST_DEVICE constexpr MatrixSpan(T *data, std::size_t num_rows,
                                           std::size_t num_cols,
                                           std::size_t pitch)
      : data_(data), num_rows_(num_rows), num_cols_(num_cols), pitch_(pitch) {
    assert(data);
    assert(num_rows > 0);
    assert(num_cols > 0);
    assert(pitch >= num_cols);
  }

  SIMT_TF_HOST_DEVICE constexpr RowSpan<T> operator[](std::size_t index) const
      noexcept {
    assert(index < num_rows_);

    return {data_ + index * pitch_, num_cols_};
  }

  SIMT_TF_HOST_DEVICE constexpr std::size_t num_rows() const noexcept {
    return num_rows_;
  }

  SIMT_TF_HOST_DEVICE constexpr std::size_t num_cols() const noexcept {
    return num_cols_;
  }

  SIMT_TF_HOST_DEVICE constexpr std::size_t pitch() const noexcept {
    return pitch_;
  }

  SIMT_TF_HOST_DEVICE constexpr std::size_t num_elements() const noexcept {
    return num_rows_ * num_cols_;
  }

private:
  T *data_;
  std::size_t num_rows_;
  std::size_t num_cols_;
  std::size_t pitch_; // number of elements between rows
};

/**
 *  A view into a row of a matrix.
 */
template <typename T> class RowSpan {
public:
  SIMT_TF_HOST_DEVICE constexpr RowSpan(T *data, std::size_t num_cols) noexcept
      : data_(data), num_cols_(num_cols) {
    assert(data);
    assert(num_cols > 0);
  }

  SIMT_TF_HOST_DEVICE constexpr T &operator[](std::size_t index) const
      noexcept {
    assert(index < num_cols_);

    return data_[index];
  }

  constexpr std::size_t size() const noexcept { return num_cols_; }

private:
  T *data_;
  std::size_t num_cols_;
};

} // namespace simt_tf

#endif
