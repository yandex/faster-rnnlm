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
//    GetWeights returns a pointer to class weights (derived from IRecWeights)
//
// IRecWeights stands for anything that could Dump to file and Load from file
// It also must have function DiagonalInitialization to mimic initialization
// from http://arxiv.org/pdf/1504.00941v2.pdf
//
// Updaters (objects that could use weights to perform propagation)
// - has 4 accessor methods returning state-per-timestamp matrices:
//     GetInputMatrix()
//       returns write-only matrix where user have to put previous hidden states (inputs)
//     GetOutputMatrix()
//       returns read-only with outputs of the reccurent layer
//     GetOutputGradMatrix()
//       returns write-only matrix where user have to put gradients
//       with respect to output layer
//     GetInputGradMatrix()
//       returns read-only matrix with gradients over the GetInputMatrix()
//
// - must define three action methods:
//   ForwardSequence
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

  virtual void Dump(FILE* fo) const {
    for (size_t i = 0; i < matrices_.size(); ++i) {
      ::Dump(matrices_[i], fo);
    }
    for (size_t i = 0; i < vectors_.size(); ++i) {
      ::Dump(vectors_[i], fo);
    }
  }

  virtual void Load(FILE* fo) {
    for (size_t i = 0; i < matrices_.size(); ++i) {
      ::Load(&matrices_[i], fo);
    }
    for (size_t i = 0; i < vectors_.size(); ++i) {
      ::Load(&vectors_[i], fo);
    }
  }

  virtual void DiagonalInitialization(Real) = 0;

 protected:
  IRecWeights(int syn_count, int bias_count)
      : matrices_(syn_count), vectors_(bias_count) {}

  std::vector<RowMatrix> matrices_;
  std::vector<RowVector> vectors_;
};

#endif  // FASTER_RNNLM_LAYERS_INTERFACE_H_

