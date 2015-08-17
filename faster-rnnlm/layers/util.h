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
    TUpdater* updater = static_cast<TUpdater*>(this);
    for (int step = steps - 1; step > 0; --step, --bptt_phase) {
      updater->BackwardStep(step);
      if (bptt == 0 || (bptt_phase >= (bptt_period - bptt))) {
        updater->BackwardStepThroughTime(step);
      }
      if (bptt_phase == 0) {
        bptt_phase = bptt_period;
      }
    }
    if (steps != 0) {
      updater->BackwardStep(0);
    }
  }

  // Back propagate all connections within single timestamp
  // Must be defined in each derived class
  void BackwardStep(int step) { exit(1); }

  // Back propagate hidden_{step - 1} -> hidden_{step} (step > 0)
  // Must be defined in each derived class
  void BackwardStepThroughTime(int step) { exit(1); }
};

#endif  // FASTER_RNNLM_LAYERS_UTIL_H_
