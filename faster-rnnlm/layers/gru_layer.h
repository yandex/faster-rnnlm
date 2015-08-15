#ifndef FASTER_RNNLM_LAYERS_GRU_LAYER_H_
#define FASTER_RNNLM_LAYERS_GRU_LAYER_H_

#include <stdio.h>

#include "faster-rnnlm/layers/interface.h"
#include "faster-rnnlm/layers/util.h"
#include "faster-rnnlm/settings.h"
#include "faster-rnnlm/util.h"

// Gated Recurrent Unit
// http://arxiv.org/pdf/1412.3555.pdf
//
// The unit could be defined using the following parameters:
//   x_{t} = input_.row(t)          - input vector
//   u_{t} = update_.row(t)         - update gate
//   r_{t} = reset_.row(t)          - reset gate
//   p_{t} = partialhidden_.row(t)  - previous hidden after reset
//   q_{t} = quasihidden_.row(t)    - hidden state candidate
//   h_{t} = output.row(t)          - actual hidden state (output vector)
//
// Matrices:
//   I^{u} = syn_update_in_          W^{u} = syn_update_out_
//   I^{r} = syn_reset_in_           W^{r} = syn_reset_out_
//   I^{q} = syn_quasihidden_in_     W^{q} = syn_quasihidden_out_
// Vectors:
//   b^{u} = bias_update_
//   b^{r} = bias_reset_
//
// The following constrains hold:
//  h_{-1} = 0 (vector of zeros)
//  u_{t} = \sigma(I^u x_t + W^u h_{t - 1} + b^u)
//  r_{t} = \sigma(I^u x_t + W^r h_{t - 1} + b^r)
//  p_{t} = r_t .*. h_{t - 1}   (elementwise multiplication)
//  q_{t} = tanh(I^q x_t + W^q p_{t})
//  h_{t} = h_{t - 1} .*. (1 - u_t) +  q_{t} .*. u_t

class GRULayer : public IRecLayer {
 public:
  class Updater;

  class Weights : public IStdWeights {
   public:
    explicit Weights(int layer_size)
        :IStdWeights(6, 2, layer_size)
        , syn_reset_in_(matrices_[0])
        , syn_reset_out_(matrices_[1])
        , syn_update_in_(matrices_[2])
        , syn_update_out_(matrices_[3])
        , syn_quasihidden_in_(matrices_[4])
        , syn_quasihidden_out_(matrices_[5])
        , bias_reset_(vectors_[0])
        , bias_update_(vectors_[1])
    {
      // initialize input synapses with identity matrices
      syn_reset_in_.setIdentity();
      syn_update_in_.setIdentity();
      syn_quasihidden_in_.setIdentity();
    }

    void DiagonalInitialization(Real alpha) {}

   protected:
    RowMatrix& syn_reset_in_;
    RowMatrix& syn_reset_out_;
    RowMatrix& syn_update_in_;
    RowMatrix& syn_update_out_;
    RowMatrix& syn_quasihidden_in_;
    RowMatrix& syn_quasihidden_out_;

    RowVector& bias_reset_;
    RowVector& bias_update_;

    friend class Updater;
  };

  GRULayer(int layer_size, bool use_bias, bool use_input_weights)
    : weights_(layer_size), use_bias_(use_bias) , use_input_weights_(use_input_weights) {}

  IRecUpdater* CreateUpdater();

  IRecWeights* GetWeights() { return &weights_; }

 private:
  friend class Updater;

  Weights weights_;
  const bool use_bias_;
  const bool use_input_weights_;
};

#endif  // FASTER_RNNLM_LAYERS_GRU_LAYER_H_


