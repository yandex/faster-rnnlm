#ifndef FASTER_RNNLM_RECURRENT_H_
#define FASTER_RNNLM_RECURRENT_H_

#include <inttypes.h>
#include <stdio.h>

#include <cmath>
#include <string>

#include "faster-rnnlm/settings.h"
#include "faster-rnnlm/util.h"

class IRecLayer;

// factory to create layer of given type
// returns NULL if layer_type is invalid
IRecLayer* CreateLayer(const std::string& layer_type, int layer_size, int layer_count);


// ============== ACTIVATION FUNCTIONS ================================
//
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


// ============== RECURRENT LAYERS ================================
//
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
  const RowMatrix& GetInputGradMatrix() const { return input_g_; }
  const RowMatrix& GetOutputMatrix() const { return output_; }
  RowMatrix& GetOutputGradMatrix() { return output_g_; }

  void ForwardSequence(int steps) { return ForwardSubSequence(0, steps); }
  void ForwardStep(int step_idx) { return ForwardSubSequence(step_idx, 1); }

  virtual void BackwardSequence(int steps, uint32_t truncation_seed, int bptt_period, int bptt) = 0;
  virtual void UpdateWeights(int steps, Real lrate, Real l2reg, Real rmsprop) = 0;
  virtual void ForwardSubSequence(int start, int steps) = 0;

 protected:
  const int size_;

  RowMatrix input_, input_g_;
  RowMatrix output_, output_g_;
};


class IRecWeights {
 public:
  virtual ~IRecWeights() {}

  virtual void Dump(FILE* fo) const = 0;

  virtual void Load(FILE* fo) = 0;

  virtual void DiagonalInitialization(Real) = 0;
};


// IStdWeights stores a few square matrices and a few bias vectors
// It can dump and load them from a file object
template<int kSynCount, int kBiasCount = 0>
class IStdWeights : public IRecWeights {
 public:
  explicit IStdWeights(int layer_size) : layer_size_(layer_size) {
    for (int i = 0; i < kSynCount; ++i) {
      syns_[i].resize(layer_size_, layer_size_);
      if (layer_size_) {
        InitNormal(1. / std::sqrt(layer_size_), &syns_[i]);
      }
    }
    for (int i = 0; i < kBiasCount; ++i) {
      biases_[i].resize(1, layer_size_);
      biases_[i].setZero();
    }
  }

  virtual void Dump(FILE* fo) const {
    for (int i = 0; i < kSynCount; ++i) {
      ::Dump(syns_[i], fo);
    }
    for (int i = 0; i < kBiasCount; ++i) {
      ::Dump(biases_[i], fo);
    }
  }

  virtual void Load(FILE* fo) {
    for (int i = 0; i < kSynCount; ++i) {
      ::Load(&syns_[i], fo);
    }
    for (int i = 0; i < kBiasCount; ++i) {
      ::Load(&biases_[i], fo);
    }
  }

 protected:
  const int layer_size_;
  RowMatrix syns_[kSynCount];
  RowVector biases_[kBiasCount];
};


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

  class Weights : public IStdWeights<2> {
   public:
    Weights(int layer_size, bool use_input_weights)
      : IStdWeights<2>(layer_size), syn_rec_(syns_[0]), syn_in_(syns_[1])
        , use_input_weights_(use_input_weights) {}

    void Dump(FILE* fo) const;
    void Load(FILE* fo);

    void DiagonalInitialization(Real alpha);

   protected:
    RowMatrix& syn_rec_;
    RowMatrix& syn_in_;
    const bool use_input_weights_;
    friend class Updater;
  };


  SimpleRecurrentLayer(int layer_size, bool use_input_weights, IActivation* activation)
    : weights_(layer_size, use_input_weights)
    , use_input_weights_(use_input_weights)
    , activation_(activation) {}

  ~SimpleRecurrentLayer() { delete activation_; }

  IRecUpdater* CreateUpdater();

  IRecWeights* GetWeights() { return &weights_; }

 private:
  Weights weights_;
  const bool use_input_weights_;
  IActivation* activation_;

  friend class Updater;
};


// Gated Recurrent Unit
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

  class Weights : public IStdWeights<6, 2> {
   public:
    explicit Weights(int layer_size)
        :IStdWeights<6, 2>(layer_size)
        , syn_reset_in_(syns_[0])
        , syn_reset_out_(syns_[1])
        , syn_update_in_(syns_[2])
        , syn_update_out_(syns_[3])
        , syn_quasihidden_in_(syns_[4])
        , syn_quasihidden_out_(syns_[5])
        , bias_reset_(biases_[0])
        , bias_update_(biases_[1])
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
  const int context_size_;
  const bool use_input_weights_;
};

#endif  // FASTER_RNNLM_RECURRENT_H_
