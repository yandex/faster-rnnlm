#ifndef FASTER_RNNLM_LAYERS_SCRN_LAYER_H_
#define FASTER_RNNLM_LAYERS_SCRN_LAYER_H_

#include "faster-rnnlm/layers/interface.h"
#include "faster-rnnlm/settings.h"
#include "faster-rnnlm/util.h"


// Structurally Constrained Recurrent Network
// http://arxiv.org/pdf/1412.7753v2.pdf
//
// Let context_size, hidden_size, and input_size be some integers
// such that context_size + hidden_size = input_size
//
// The unit could be defined using the following parameters:
//   x_{t} = input_.row(t)     - input vector   (input_size)
//   s_{t} = context_.row(t)   - context vector (context_size)
//   h_{t} = hidden_.row(t)    - hidden vector  (hidden_size)
//   o_{t} = output_.row(t)    - output vector  (input_size)
//
// Matrices:
//   B = syn_context_        context_size X input_size
//   P = syn_rec_context_    hidden_size  X context_size
//   A = syn_rec_input_      hidden_size  X input_size
//   R = syn_rec_hidden_     hidden_size  X hidden_size
// Vectors:
//   a = decay_context_      context_size
//
// The following constrains hold:
//  s_{-1} = 0 (vector of zeros)
//  h_{-1} = 0 (vector of zeros)
//  s_{t}  = (1 - a) .*. (B x_{t}) - a .*. s_{t-1}
//     .*. stands for elementwise multiplication
//  h_{t}  = \sigma(P s_{t} + A x_{t} + R h_{t - 1})
//  o_{t}  = concatenate(s_{t}; h_{t})

class SCRNLayer : public IRecLayer {
 public:
  class Updater;

  class Weights;

  // Note, that context_size must be less than or equal to layer_size
  SCRNLayer(int layer_size, int context_size, bool use_input_weights);

  ~SCRNLayer();

  IRecUpdater* CreateUpdater();

  IRecWeights* GetWeights();

 private:
  friend class Updater;

  Weights* weights_;
  const bool use_input_weights_;
};

#endif  // FASTER_RNNLM_LAYERS_SCRN_LAYER_H_

