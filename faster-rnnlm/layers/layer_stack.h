#ifndef FASTER_RNNLM_LAYERS_LAYER_STACK_H_
#define FASTER_RNNLM_LAYERS_LAYER_STACK_H_

#include <inttypes.h>
#include <vector>

#include "faster-rnnlm/layers/interface.h"
#include "faster-rnnlm/settings.h"
#include "faster-rnnlm/util.h"


struct LayerStack : public IRecLayer {
  struct Updater : public IRecUpdater {
    explicit Updater(LayerStack* stack) : IRecUpdater(stack->layer_size) {
      for (size_t i = 0; i < stack->layers.size(); ++i) {
        updaters.push_back(stack->layers[i]->CreateUpdater());
      }
    }

    ~Updater() {
      for (size_t i = 0; i < updaters.size(); ++i) {
        delete updaters[i];
      }
    }

    virtual void BackwardSequence(int steps, uint32_t truncation_seed, int bptt_period, int bptt) {
      updaters[updaters.size() - 1]->GetOutputGradMatrix().topRows(steps)
        = GetOutputGradMatrix().topRows(steps);
      for (size_t i = updaters.size(); i --> 0; ) {
        updaters[i]->BackwardSequence(steps, truncation_seed, bptt_period, bptt);
        if (i > 0) {
          updaters[i - 1]->GetOutputGradMatrix().topRows(steps)
            = updaters[i]->GetInputGradMatrix().topRows(steps);
        }
      }
      input_g_.topRows(steps) = updaters[0]->GetInputGradMatrix().topRows(steps);
    }

    virtual void UpdateWeights(
        int steps, Real lrate, Real l2reg, Real rmsprop, Real gradient_clipping) {
      for (size_t i = 0; i < updaters.size(); ++i) {
        updaters[i]->UpdateWeights(steps, lrate, l2reg, rmsprop, gradient_clipping);
      }
    }

    virtual void ForwardSubSequence(int start, int steps) {
      updaters[0]->GetInputMatrix().middleRows(start, steps) = input_.middleRows(start, steps);
      for (size_t i = 0; i < updaters.size(); ++i) {
        updaters[i]->ForwardSubSequence(start, steps);
        if (i < updaters.size() - 1) {
          updaters[i + 1]->GetInputMatrix().middleRows(start, steps)
            = updaters[i]->GetOutputMatrix().middleRows(start, steps);
        }
      }
      output_.middleRows(start, steps)
        = updaters[updaters.size() - 1]->GetOutputMatrix().middleRows(start, steps);
    }

    std::vector<WeightMatrixUpdater<RowMatrix>*> GetMatrices() {
      std::vector<WeightMatrixUpdater<RowMatrix>*> v, layer_v;
      for (size_t i = 0; i < updaters.size(); ++i) {
        layer_v = updaters[i]->GetMatrices();
        v.insert(v.end(), layer_v.begin(), layer_v.end());
      }
      return v;
    }

    std::vector<WeightMatrixUpdater<RowVector>*> GetVectors() {
      std::vector<WeightMatrixUpdater<RowVector>*> v, layer_v;
      for (size_t i = 0; i < updaters.size(); ++i) {
        layer_v = updaters[i]->GetVectors();
        v.insert(v.end(), layer_v.begin(), layer_v.end());
      }
      return v;
    }

    std::vector<IRecUpdater*> updaters;
  };

  struct Weights : public IRecWeights {
    explicit Weights(LayerStack* stack) : IRecWeights(0, 0), stack(stack) {}

    virtual std::vector<RowMatrix*> GetMatrices() {
      std::vector<RowMatrix*> v, layer_v;
      for (size_t i = 0; i < stack->layers.size(); ++i) {
        layer_v = stack->layers[i]->GetWeights()->GetMatrices();
        v.insert(v.end(), layer_v.begin(), layer_v.end());
      }
      return v;
    }

    virtual std::vector<RowVector*> GetVectors() {
      std::vector<RowVector*> v, layer_v;
      for (size_t i = 0; i < stack->layers.size(); ++i) {
        layer_v = stack->layers[i]->GetWeights()->GetVectors();
        v.insert(v.end(), layer_v.begin(), layer_v.end());
      }
      return v;
    }

    void DiagonalInitialization(Real x) {
      for (size_t i = 0; i < stack->layers.size(); ++i) {
        stack->layers[i]->GetWeights()->DiagonalInitialization(x);
      }
    }

    LayerStack* stack;
  };

  LayerStack(int layer_size, const std::vector<IRecLayer*>& layers)
    : layer_size(layer_size), layers(layers), weights(this) {}

  ~LayerStack() {
    for (size_t i = 0; i < layers.size(); ++i) {
      delete layers[i];
    }
  }

  IRecUpdater* CreateUpdater() { return new Updater(this); }

  IRecWeights* GetWeights() { return &weights; }

  int layer_size;
  std::vector<IRecLayer*> layers;
  Weights weights;
};

#endif  // FASTER_RNNLM_LAYERS_LAYER_STACK_H_

