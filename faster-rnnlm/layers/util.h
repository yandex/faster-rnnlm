#ifndef FASTER_RNNLM_LAYERS_UTIL_H_
#define FASTER_RNNLM_LAYERS_UTIL_H_

#include "faster-rnnlm/layers/interface.h"
#include "faster-rnnlm/settings.h"
#include "faster-rnnlm/util.h"

// IStdWeights assumes that all matrices have size layer_size X layer_size
// and bias vectors have size layer_size X 1
//
// Matrix weights are unitialized with normal random values
class IStdWeights : public IRecWeights {
 public:
  IStdWeights(int syn_count, int bias_count, int layer_size)
      : IRecWeights(syn_count, bias_count), layer_size_(layer_size) {

    for (size_t i = 0; i < matrices_.size(); ++i) {
      matrices_[i].resize(layer_size_, layer_size_);
      if (layer_size_) {
        InitNormal(1. / std::sqrt(layer_size_), &matrices_[i]);
      }
    }
    for (size_t i = 0; i < vectors_.size(); ++i) {
      vectors_[i].resize(1, layer_size_);
      vectors_[i].setZero();
    }
  }

 protected:
  const int layer_size_;
};

// Helper function to update weights of a linear component within a recurrent layer
//
// Linear components looks like F_t(x) = A * x_t, where A is matrix and x is a vector
// per_step_values[t] contains value of x_t for timestamp t
// per_step_value_grads[t] contains value of dL/dF_t for timestamp t (L is a loss function)
//
// per_step_values and per_step_value_grads are assumed to have at least steps rows
//
// Matrix templates are required to handle cases, when per_step matrices are actually submatrices
template<class Matrix1, class Matrix2>
void CalculateRecurrentWeightGradients(
    int steps, const Matrix1& per_step_values, const Matrix2& per_step_value_grads,
    RowMatrix* weight_grads) {
  if (steps == 0) {
    weight_grads->setZero();
    return;
  }

  weight_grads->noalias() = (
      per_step_value_grads.topRows(steps).transpose()
      * per_step_values.topRows(steps));

  (*weight_grads) /= (steps + 1);
}

// Helper function to update weights of a linear component within a recurrent layer
//
// Linear components looks like F_t(x) = A * x_t, where A is matrix and x is a vector
// per_step_values[t] contains value of x_t for timestamp t
// per_step_value_grads[t] contains value of dL/dF_t for timestamp t (L is a loss function)
//
// per_step_values and per_step_value_grads are assumed to have at least steps rows
//
// weight_grads is a matrix that is used to accumulate gradients of dL/dA
// weight_mean_grads is a matrix that accumulates weight grads to apply rmsprop
// weights is the weight matrix, i.e. 'A'

// Matrix templates are required to handle cases, when per_step matrices are actually submatrices
template<class Matrix1, class Matrix2, class Matrix3>
void UpdateRecurrentSynWeights(
    int steps, Real lrate, Real l2reg, Real rmsprop, Real gradient_clipping,
    const Matrix1& per_step_values, const Matrix2& per_step_value_grads,
    WeightMatrixUpdater<Matrix3>* updater) {
  if (steps == 0) {
    return;
  }

  updater->GetGradients()->noalias() = (
      per_step_value_grads.topRows(steps).transpose()
      * per_step_values.topRows(steps));

  (*updater->GetGradients()) /= (steps + 1);

  updater->ApplyGradients(lrate, l2reg, rmsprop, gradient_clipping);
}


// Curiously recurring template pattern
template<class TUpdater>
class TruncatedBPTTMixin {
 protected:
  // Truncated Backpropagation in time
  //
  // truncation_seed - seed that determines BPTT unfoliding start
  // bptt_period - period of BPTT
  // bptt - number of steps for real through time propagation;
  //   if bptt = 0, no truncation is performed
  //   otherwise bptt must be less or equal to bptt_period
  void BackwardSequenceTruncated(int steps, uint32_t truncation_seed, int bptt_period, int bptt) {
    if (bptt == 0) {
      bptt_period = 1;
    }
    int bptt_phase = (truncation_seed + bptt_period - 1) % bptt_period;
    for (int step = steps - 1; step > 0; --step, --bptt_phase) {
      static_cast<TUpdater*>(this)->BackwardStep(step);
      if (bptt == 0 || (bptt_phase >= (bptt_period - bptt))) {
        static_cast<TUpdater*>(this)->BackwardStepThroughTime(step);
      }
      if (bptt_phase == 0) {
        bptt_phase = bptt_period;
      }
    }
    if (steps != 0) {
      static_cast<TUpdater*>(this)->BackwardStep(0);
    }
  }

  // Back prapogate everything except for hiden->hidden
  // Must be defined in derived class
  void BackwardStep(int step) { exit(1); }

  // Back propagate hidden_{step - 1} -> hidden_{step} (step > 0)
  // Must be defined in derived class
  void BackwardStepThroughTime(int step) { exit(1); }
};

#endif  // FASTER_RNNLM_LAYERS_UTIL_H_
