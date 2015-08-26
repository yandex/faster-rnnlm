#ifndef FASTER_RNNLM_LAYERS_SIMPLE_LAYER_H_
#define FASTER_RNNLM_LAYERS_SIMPLE_LAYER_H_

#include <stdio.h>

#include "faster-rnnlm/layers/util.h"
#include "faster-rnnlm/settings.h"
#include "faster-rnnlm/util.h"


struct IActivation;

// Simple recurrent layer
//
// The layer could be described using the following variables:
//   x_{t} = input_.row(t)          - input vector
//   h_{t} = output.row(t)          - hidden state (output vector)
//
//   f(x)  = activation_->Forward   - activation function
//
// Matrices:
//   V = syn_in_                    - input transformation weights (optional)
//   W = syn_rec_                   - recurrent weights
//
//
// The following constrains hold:
//  h_{-1} = 0 (vector of zeros)
//  h_{t} = f(V * x_{t} + W * h_{t - 1})
class SimpleRecurrentLayer : public IRecLayer {
 public:
  class Updater;

  class Weights : public IStdWeights {
   public:
    Weights(int layer_size, bool use_input_weights)
      : IStdWeights(2, 0, layer_size), syn_rec_(matrices_[0]), syn_in_(matrices_[1])
        , use_input_weights_(use_input_weights) {}

    std::vector<RowMatrix*> GetMatrices();

    void DiagonalInitialization(Real alpha);

   protected:
    RowMatrix& syn_rec_;
    RowMatrix& syn_in_;
    const bool use_input_weights_;
    friend class Updater;
  };


  // Note, that the layer become an owner of the activation function
  SimpleRecurrentLayer(int layer_size, bool use_input_weights, IActivation* activation)
    : weights_(layer_size, use_input_weights)
    , use_input_weights_(use_input_weights)
    , activation_(activation) {}

  ~SimpleRecurrentLayer();

  IRecUpdater* CreateUpdater();

  IRecWeights* GetWeights() { return &weights_; }

 private:
  Weights weights_;
  const bool use_input_weights_;
  IActivation* activation_;

  friend class Updater;
};

#endif  // FASTER_RNNLM_LAYERS_SIMPLE_LAYER_H_

