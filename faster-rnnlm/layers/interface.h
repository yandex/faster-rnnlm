#ifndef FASTER_RNNLM_LAYERS_INTERFACE_H_
#define FASTER_RNNLM_LAYERS_INTERFACE_H_

#include <stdio.h>

#include <string>
#include <vector>

#include "faster-rnnlm/settings.h"
#include "faster-rnnlm/util.h"

class IRecLayer;

// factory to create layer of given type
// returns NULL if layer_type is invalid
IRecLayer* CreateLayer(const std::string& layer_type, int layer_size, int layer_count);


// Each layer must derive from IRecLayer and define two functions
//    CreateUpdater returns a new instance of a class derived from IRecUpdater
//    GetWeights returns a pointer to class with weights (derived from IRecWeights)
//
// IRecWeights stands for anything that could Dump to file and Load from file
// It also must define function DiagonalInitialization to mimic initialization
// from http://arxiv.org/pdf/1504.00941v2.pdf
// If such initialization is not applicable the the method should do nothing
//
// Updaters (objects that could use weights to perform propagation)
// - has 4 accessor methods returning state-per-timestamp matrices:
//     GetInputMatrix()
//       returns matrix where user have to put previous hidden states (inputs)
//     GetOutputMatrix()
//       returns with outputs of the reccurent layer
//     GetOutputGradMatrix()
//       returns matrix where user have to put gradients
//       with respect to output layer
//     GetInputGradMatrix()
//       returns matrix with gradients over the GetInputMatrix()
//
// - must define three action methods:
//   ForwardSubSequence
//     makes a forward pass over given number of steps over GetInputMatrix() matrix
//     fills GetOutputMatrix() matrix with per-step outputs
//   BackwardSequence
//     makes BPTT
//     assumes that GetOutputGradMatrix() is filled with loss function gradients
//     with respect to output
//     fills GetInputGradMatrix() with loss function gradients with respect to input
//   UpdateWeights
//     updates layer weights
//
// - one should create a separate updater within each thread using layer's CreateUpdater method
class IRecUpdater;
class IRecWeights;

class IRecLayer {
 public:
  virtual IRecUpdater* CreateUpdater() = 0;
  virtual IRecWeights* GetWeights() = 0;
  virtual ~IRecLayer() {}
};


class IRecUpdater {
 public:
  IRecUpdater(int layer_size)
    : size_(layer_size)
    , input_(MAX_SENTENCE_WORDS, size_)
    , input_g_(MAX_SENTENCE_WORDS, size_)
    , output_(MAX_SENTENCE_WORDS, size_)
    , output_g_(MAX_SENTENCE_WORDS, size_) {}

  virtual ~IRecUpdater() {}

  RowMatrix& GetInputMatrix() { return input_; }
  RowMatrix& GetInputGradMatrix() { return input_g_; }
  RowMatrix& GetOutputMatrix() { return output_; }
  RowMatrix& GetOutputGradMatrix() { return output_g_; }

  void ForwardSequence(int steps) { return ForwardSubSequence(0, steps); }
  void ForwardStep(int step_idx) { return ForwardSubSequence(step_idx, 1); }

  virtual void BackwardSequence(int steps, uint32_t truncation_seed, int bptt_period, int bptt) = 0;
  virtual void UpdateWeights(int steps, Real lrate, Real l2reg, Real rmsprop, Real gradient_clipping) = 0;
  virtual void ForwardSubSequence(int start, int steps) = 0;

  // Returns list of pointers on updates
  // The order must much one in corresponding methods in weight class
  virtual std::vector<WeightMatrixUpdater<RowMatrix>*> GetMatrices() = 0;
  virtual std::vector<WeightMatrixUpdater<RowVector>*> GetVectors() = 0;

 protected:
  const int size_;

  RowMatrix input_, input_g_;
  RowMatrix output_, output_g_;
};


// IRecWeights stores a few square matrices and a few bias vectors
// It can dump and load them from a file object
class IRecWeights {
 public:
  virtual ~IRecWeights() {}

  virtual void DiagonalInitialization(Real) = 0;

  // Returns list of pointers on weight matrices
  virtual std::vector<RowMatrix*> GetMatrices() {
    std::vector<RowMatrix*> v;
    for (size_t i = 0; i < matrices_.size(); ++i) {
      v.push_back(&matrices_[i]);
    }
    return v;
  }

  // Returns list of pointers on weight vectors
  virtual std::vector<RowVector*> GetVectors() {
    std::vector<RowVector*> v;
    for (size_t i = 0; i < vectors_.size(); ++i) {
      v.push_back(&vectors_[i]);
    }
    return v;
  }

  void Dump(FILE* fo) {
    DumpMatrixArray(GetMatrices(), fo);
    DumpMatrixArray(GetVectors(), fo);
  }

  void Load(FILE* fo) {
    LoadMatrixArray(GetMatrices(), fo);
    LoadMatrixArray(GetVectors(), fo);
  }

 protected:
  IRecWeights(int syn_count, int bias_count)
      : matrices_(syn_count), vectors_(bias_count) {}

  std::vector<RowMatrix> matrices_;
  std::vector<RowVector> vectors_;
};

#endif  // FASTER_RNNLM_LAYERS_INTERFACE_H_

