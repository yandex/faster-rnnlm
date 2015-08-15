#ifndef FASTER_RNNLM_LAYERS_SCRN_LAYER_H_
#define FASTER_RNNLM_LAYERS_SCRN_LAYER_H_

#include <inttypes.h>
#include <stdio.h>

#include "faster-rnnlm/layers/activation_functions.h"
#include "faster-rnnlm/layers/interface.h"
#include "faster-rnnlm/settings.h"
#include "faster-rnnlm/util.h"


// Structurally Constrained Recurrent Network
//
// http://arxiv.org/pdf/1412.7753v2.pdf
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

