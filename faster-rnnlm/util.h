#ifndef FASTER_RNNLM_UTIL_H_
#define FASTER_RNNLM_UTIL_H_

#include <inttypes.h>
#include <stdio.h>

#include <eigen3/Eigen/Dense>

#include "faster-rnnlm/settings.h"

using Eigen::Ref;

typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;
typedef Eigen::Matrix<Real, 1, Eigen::Dynamic, Eigen::RowMajor> RowVector;


inline void FreadAllOrDie(void* ptr, size_t size, size_t count, FILE* fo, const char* message) {
  size_t read = fread(ptr, size, count, fo);
  if (read != count) {
    fprintf(
        stderr, "ERROR: expected to read %zu elements, but read %zu elements (%s)\n",
        count, read, message);
    exit(1);
  }
}


template<int rows, int _MaxRows=rows, int _MaxCols = -1>
inline void Dump(const Eigen::Matrix<Real, rows, Eigen::Dynamic, Eigen::RowMajor>& matrix, FILE* fo) {
  fwrite(matrix.data(), sizeof(Real), matrix.rows() * matrix.cols(), fo);
}


template<int rows, int _MaxRows = rows, int _MaxCols = -1>
inline void Load(Eigen::Matrix<Real, rows, Eigen::Dynamic, Eigen::RowMajor, _MaxRows, _MaxCols>* matrix, FILE* fo) {
  FreadAllOrDie(
      matrix->data(), sizeof(Real), matrix->rows() * matrix->cols(), fo,
      "failed to read matrix");
}


template<class Matrix>
void DumpMatrixArray(std::vector<Matrix*> array, FILE* fo) {
  for (size_t i = 0; i < array.size(); ++i) {
    Dump(*array[i], fo);
  }
}


template<class Matrix>
void LoadMatrixArray(std::vector<Matrix*> array, FILE* fo) {
  for (size_t i = 0; i < array.size(); ++i) {
    Load(array[i], fo);
  }
}


inline void InitUniform(Real max_value, size_t sz, Real* weights) {
  for (size_t i = 0; i < sz; i++) {
    weights[i] = (rand() / static_cast<Real>(RAND_MAX) - 0.5) * 2 * max_value;
  }
}


inline void InitNormal(Real stddev, size_t sz, Real* weights) {
  for (size_t i = 0; i < sz; i++) {
    Real standard_normal = -6;
    for (int j = 0; j < 12; ++j) {
      standard_normal += rand() / static_cast<Real>(RAND_MAX);
    }
    weights[i] = standard_normal * stddev;
  }
}


template<class Matrix>
inline void InitUniform(Real max_value, Matrix* weights) {
  InitUniform(max_value, weights->cols() * weights->rows(), weights->data());
}


template<class Matrix>
inline void InitNormal(Real stddev, Matrix* weights) {
  InitNormal(stddev, weights->cols() * weights->rows(), weights->data());
}


inline Real Clip(Real x, Real max_abs) {
  x = (x < 0 || x > 0) ? x : 0;  // replace nan with 0
  x = (x < max_abs) ? x : max_abs;
  x = (x > -max_abs) ? x : -max_abs;
  return x;
}


inline void ClipMatrix(Eigen::Ref<RowMatrix> matrix, Real max_abs) {
  size_t size = matrix.cols() * matrix.rows();
  for (size_t i = 0; i < size; ++i) {
    matrix.data()[i] = Clip(matrix.data()[i], max_abs);
  }
}


inline void ShrinkMatrix(Eigen::Ref<RowMatrix> matrix, Real max_abs) {
  const Real huge = max_abs * 1e4;
  size_t size = matrix.cols() * matrix.rows();
  Real max_value = 0;
  for (size_t i = 0; i < size; ++i) {
    Real& x = matrix.data()[i];

    // NaN -> zero
    x = (x < 0 || x > 0) ? x : 0;

    // inf -> huge
    x = (x > huge) ? huge : x;
    x = (x < -huge) ? -huge : x;

    Real x_abs = (x > 0) ? x : -x;
    max_value = (x_abs > max_value) ? x_abs : max_value;
  }
  if (max_value > max_abs) {
    matrix /= max_value;
  }
}


// The class stores and applies gradients of some network paramaters
template<class Matrix>
class WeightMatrixUpdater {
 public:
  explicit WeightMatrixUpdater(Matrix* weights)
    : weights_(*weights)
    , gradients_(weights_.rows(), weights_.cols())
    , mean_squared_gradients_(weights_.rows(), weights_.cols())
  {
    gradients_.setZero();
    mean_squared_gradients_.setConstant(1);
  }

  const Matrix& GetWeights() const { return weights_; }
  const Matrix& W() const { return GetWeights(); }

  Matrix* GetGradients() { return &gradients_; }

  // user must fill gradients in GetWeights before ApplyGradients call
  void ApplyGradients(Real lrate, Real l2reg, Real rmsprop, Real gradient_clipping) {
    if (rmsprop >= 0) {
      mean_squared_gradients_.array() *= rmsprop;
      mean_squared_gradients_.array() += (1 - rmsprop) * gradients_.array().square();
      gradients_.array() /= mean_squared_gradients_.array().sqrt() + RMS_DAMPING_FACTOR;
    }

    ClipMatrix(gradients_, gradient_clipping);

    weights_ *= (1 - l2reg);
    weights_ += gradients_ * lrate;
  }

 protected:
  Matrix& weights_;

  Matrix gradients_;
  Matrix mean_squared_gradients_;
};

// Java 32bit random generator
inline uint32_t GetNextRandom(uint64_t* state) {
    *state = ((*state) * 25214903917ULL + 11) & ((1ULL << 48) - 1);
    uint32_t value = ((*state) >> 16);
    return value;
}

#endif  // FASTER_RNNLM_UTIL_H_
