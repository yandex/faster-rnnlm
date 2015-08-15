#include "faster-rnnlm/layers/interface.h"

#include "faster-rnnlm/layers/gru_layer.h"
#include "faster-rnnlm/layers/layer_stack.h"
#include "faster-rnnlm/layers/scrn_layer.h"
#include "faster-rnnlm/layers/simple_layer.h"


IRecLayer* CreateSingleLayer(const std::string& layer_type, int layer_size, bool first_layer) {
  // SimpleRecurrentLayer arguments: (int size, bool use_input_weights, IActivation*)
  if (layer_type == "sigmoid") {
    return new SimpleRecurrentLayer(layer_size, !first_layer, new SigmoidActivation());
  }
  if (layer_type == "tanh") {
    return new SimpleRecurrentLayer(layer_size, !first_layer, new TanhActivation());
  }
  if (layer_type == "relu") {
    return new SimpleRecurrentLayer(layer_size, !first_layer, new ReLUActivation());
  }
  if (layer_type == "relu-trunc") {
    return new SimpleRecurrentLayer(layer_size, !first_layer, new TruncatedReLUActivation());
  }
  // GRULayer arguments: (int size, bool use_bias, bool use_input_weights)
  if (layer_type == "gru") {
    return new GRULayer(layer_size, false, false);
  }
  if (layer_type == "gru-bias") {
    return new GRULayer(layer_size, true, false);
  }
  if (layer_type == "gru-insyn") {
    return new GRULayer(layer_size, false, true);
  }
  if (layer_type == "gru-full") {
    return new GRULayer(layer_size, true, true);
  }
  {
    std::string prefix = "scrn";
    std::string suffix = "fast";
    if (layer_type.substr(0, prefix.size()) == prefix) {
      bool fast = false;
      int offset = prefix.size();
      if (layer_type.substr(prefix.size(), suffix.size()) == suffix) {
        fast = true;
        offset += suffix.size();
      }
      int context_size = atoi(layer_type.substr(offset).c_str());
      if (context_size > layer_size) {
        fprintf(stderr, "WARNING (SCRNLayer) context size must less than or equal to layer size\n");
        context_size = layer_size;
      }
      return new SCRNLayer(layer_size, context_size, !fast || !first_layer);
    }
  }
  return NULL;
}


IRecLayer* CreateLayer(const std::string& layer_type, int layer_size, int layer_count) {
  if (layer_count <= 0) {
    fprintf(stderr, "ERROR layer count must be positive!\n");
    exit(1);
  }

  std::vector<IRecLayer*> layers;
  for (int i = 0; i < layer_count; ++i) {
    IRecLayer* layer = CreateSingleLayer(layer_type, layer_size, (i == 0));
    if (layer == NULL) {
      for (int j = 0; j < i; ++j) {
        delete layers[j];
      }
      return NULL;
    }
    layers.push_back(layer);
  }

  if (layer_count == 1) {
    return layers[0];
  }

  return new LayerStack(layer_size, layers);
}
