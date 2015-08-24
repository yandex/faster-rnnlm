#ifndef FASTER_RNNLM_LAYERS_ACTIVATION_FUNCTIONS_H_
#define FASTER_RNNLM_LAYERS_ACTIVATION_FUNCTIONS_H_

#include <math.h>
#include <algorithm>

#include "faster-rnnlm/settings.h"


namespace {
const Real kReLUTruncation = 20;
};


// For each activation function a class with two static methods must be defined,
// namely Forward and Backward
//
// Forward applies the function f to each elemente in the given array
// Backward takes f(x) as the first argument and multiples elements
// of the second array by f'(x)
struct IActivation {
  virtual void Forward(Real* hidden, int size) = 0;
  virtual void Backward(const Real* hidden, int size, Real* hidden_g) = 0;
  virtual ~IActivation() {}
};


struct SigmoidActivation : public IActivation {
  void Forward(Real* hidden, int size) {
    for (int i = 0; i < size; i++) {
      hidden[i] = exp(hidden[i]) / (1 + exp(hidden[i]));
    }
  }

  void Backward(const Real* hidden, int size, Real* hidden_g) {
    for (int i = 0; i < size; ++i) {
      hidden_g[i] *= hidden[i] * (1 - hidden[i]);
    }
  }
};


struct TanhActivation : public IActivation {
  void Forward(Real* hidden, int size) {
    for (int i = 0; i < size; i++) {
      hidden[i] = tanh(hidden[i]);
    }
  }

  void Backward(const Real* hidden, int size, Real* hidden_g) {
    for (int i = 0; i < size; ++i) {
      hidden_g[i] *= (1 - hidden[i] * hidden[i]);
    }
  }
};


struct ReLUActivation : public IActivation {
  void Forward(Real* hidden, int size) {
    for (int i = 0; i < size; i++) {
      hidden[i] = (hidden[i] > 0) ? hidden[i] : 0;
    }
  }

  void Backward(const Real* hidden, int size, Real* hidden_g) {
    for (int i = 0; i < size; ++i) {
      hidden_g[i] *= static_cast<int>(hidden[i] > 0);
    }
  }
};


struct TruncatedReLUActivation : public IActivation {
  void Forward(Real* hidden, int size) {
    for (int i = 0; i < size; i++) {
      hidden[i] = std::min(std::max(hidden[i], static_cast<Real>(0)), kReLUTruncation);
    }
  }

  void Backward(const Real* hidden, int size, Real* hidden_g) {
    for (int i = 0; i < size; ++i) {
      hidden_g[i] *= static_cast<int>(hidden[i] > 0 && hidden[i] < kReLUTruncation);
    }
  }
};

#endif  // FASTER_RNNLM_LAYERS_ACTIVATION_FUNCTIONS_H_
