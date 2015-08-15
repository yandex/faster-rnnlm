#include "faster-rnnlm/layers/simple_layer.h"

#include <cmath>

#include "faster-rnnlm/layers/activation_functions.h"


SimpleRecurrentLayer::~SimpleRecurrentLayer() {
  delete activation_;
}

void SimpleRecurrentLayer::Weights::Dump(FILE* fo) const {
  // for backward compability
  int real_syn_count = (use_input_weights_) ? 2 : 1;
  for (int i = 0; i < real_syn_count; ++i) {
    fwrite(matrices_[i].data(), sizeof(Real), matrices_[i].rows() * matrices_[i].cols(), fo);
  }
}

void SimpleRecurrentLayer::Weights::Load(FILE* fo) {
  // for backward compability
  int real_syn_count = (use_input_weights_) ? 2 : 1;
  for (int i = 0; i < real_syn_count; ++i) {
    fread(matrices_[i].data(), sizeof(Real), matrices_[i].rows() * matrices_[i].cols(), fo);
  }
}

void SimpleRecurrentLayer::Weights::DiagonalInitialization(Real alpha) {
  syn_rec_.setIdentity();
  syn_rec_ *= alpha;
}


class SimpleRecurrentLayer::Updater
    : public IRecUpdater, public TruncatedBPTTMixin<SimpleRecurrentLayer::Updater> {
 public:
  explicit Updater(SimpleRecurrentLayer* layer)
    : IRecUpdater(layer->weights_.layer_size_)
    , use_input_weights_(layer->use_input_weights_)
    , activation_(layer->activation_)
    , syn_rec_(&layer->weights_.syn_rec_)
    , syn_in_(&layer->weights_.syn_in_)
  {
    // empty constructor
  }

  void ForwardSubSequence(int start, int steps);

  void BackwardSequence(int steps, uint32_t truncation_seed, int bptt_period, int bptt);

  void UpdateWeights(int steps, Real lrate, Real l2reg, Real rmsprop, Real gradient_clipping);

  void BackwardStep(int step);

  void BackwardStepThroughTime(int step);

 private:
  bool use_input_weights_;
  IActivation* activation_;
  WeightMatrixUpdater<RowMatrix> syn_rec_;
  WeightMatrixUpdater<RowMatrix> syn_in_;
};


void SimpleRecurrentLayer::Updater::ForwardSubSequence(int start, int steps) {
  output_.middleRows(start, steps) = input_.middleRows(start, steps);
  if (use_input_weights_) {
    output_.middleRows(start, steps) *= syn_in_.W().transpose();
  }
  for (int step = start; step < start + steps; ++step) {
    if (step != 0) {
      output_.row(step).noalias() += output_.row(step - 1) * syn_rec_.W().transpose();
    }
    activation_->Forward(output_.row(step).data(), output_.cols());
  }
}

void SimpleRecurrentLayer::Updater::BackwardSequence(
    int steps, uint32_t truncation_seed, int bptt_period, int bptt) {
  BackwardSequenceTruncated(steps, truncation_seed, bptt_period, bptt);
  input_g_.topRows(steps) = output_g_.topRows(steps);
  if (use_input_weights_) {
    input_g_.topRows(steps) *= syn_in_.W();
  }
}

void SimpleRecurrentLayer::Updater::BackwardStep(int step) {
  activation_->Backward(output_.row(step).data(), output_.cols(), output_g_.row(step).data());
}

void SimpleRecurrentLayer::Updater::BackwardStepThroughTime(int step) {
  output_g_.row(step - 1).noalias() += output_g_.row(step) * syn_rec_.W();
}


void SimpleRecurrentLayer::Updater::UpdateWeights(
    int steps, Real lrate, Real l2reg, Real rmsprop, Real gradient_clipping) {
  if (steps <= 1 || size_ == 0) {
    return;
  }
  // output_g_[t] is a gradient with respect to h_{t} (i.e. dL/dh_{t} ),
  // but it is calculated using output[t - 1] rather then output[t]
  // To take this into account we strip the first line of the output_g_
  UpdateRecurrentSynWeights(steps - 1, lrate, l2reg, rmsprop, gradient_clipping,
      output_, output_g_.bottomRows(output_g_.rows() - 1),
      &syn_rec_);

  if (use_input_weights_) {
    UpdateRecurrentSynWeights(steps, lrate, l2reg, rmsprop, gradient_clipping,
        input_, output_g_,
        &syn_in_);
  }
}


IRecUpdater* SimpleRecurrentLayer::CreateUpdater() {
  return new Updater(this);
}

