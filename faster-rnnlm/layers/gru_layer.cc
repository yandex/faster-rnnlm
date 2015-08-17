#include "faster-rnnlm/layers/gru_layer.h"

#include "faster-rnnlm/layers/activation_functions.h"


class GRULayer::Updater : public IRecUpdater, public TruncatedBPTTMixin<GRULayer::Updater> {
 public:
  explicit Updater(GRULayer* layer)
      : IRecUpdater(layer->weights_.layer_size_)
      , layer_(*layer)
      , reset_(MAX_SENTENCE_WORDS, size_), reset_g_(reset_)
      , update_(MAX_SENTENCE_WORDS, size_), update_g_(update_)
      , partialhidden_(MAX_SENTENCE_WORDS, size_), partialhidden_g_(partialhidden_)
      , quasihidden_(MAX_SENTENCE_WORDS, size_), quasihidden_g_(quasihidden_)

      , syn_reset_in_(&layer_.weights_.syn_reset_in_)
      , syn_reset_out_(&layer_.weights_.syn_reset_out_)
      , syn_update_in_(&layer_.weights_.syn_update_in_)
      , syn_update_out_(&layer_.weights_.syn_update_out_)
      , syn_quasihidden_in_(&layer_.weights_.syn_quasihidden_in_)
      , syn_quasihidden_out_(&layer_.weights_.syn_quasihidden_out_)

      , bias_reset_(&layer_.weights_.bias_reset_)
      , bias_update_(&layer_.weights_.bias_update_)
    {
        partialhidden_.row(0).setZero();
        reset_g_.row(0).setZero();
    }

  void ForwardSubSequence(int start, int steps);

  void BackwardSequence(int steps, uint32_t truncation_seed, int bptt_period, int bptt);

  void UpdateWeights(int steps, Real lrate, Real l2reg, Real rmsprop, Real gradient_clipping);

  void BackwardStep(int step);

  void BackwardStepThroughTime(int step);

  std::vector<WeightMatrixUpdater<RowMatrix>*> GetMatrices();
  std::vector<WeightMatrixUpdater<RowVector>*> GetVectors();

 private:
  GRULayer& layer_;

  RowMatrix reset_, reset_g_;
  RowMatrix update_, update_g_;
  RowMatrix partialhidden_, partialhidden_g_;
  RowMatrix quasihidden_, quasihidden_g_;

  WeightMatrixUpdater<RowMatrix> syn_reset_in_;
  WeightMatrixUpdater<RowMatrix> syn_reset_out_;
  WeightMatrixUpdater<RowMatrix> syn_update_in_;
  WeightMatrixUpdater<RowMatrix> syn_update_out_;
  WeightMatrixUpdater<RowMatrix> syn_quasihidden_in_;
  WeightMatrixUpdater<RowMatrix> syn_quasihidden_out_;

  WeightMatrixUpdater<RowVector> bias_reset_;
  WeightMatrixUpdater<RowVector> bias_update_;
};

void GRULayer::Updater::ForwardSubSequence(int start, int steps) {
  reset_.middleRows(start, steps) = input_.middleRows(start, steps);
  update_.middleRows(start, steps) = input_.middleRows(start, steps);
  quasihidden_.middleRows(start, steps) = input_.middleRows(start, steps);

  if (layer_.use_input_weights_) {
    reset_.middleRows(start, steps) *= syn_reset_in_.W().transpose();
    update_.middleRows(start, steps) *= syn_update_in_.W().transpose();
    quasihidden_.middleRows(start, steps) *= syn_quasihidden_in_.W().transpose();
  }

  for (int step = start; step < start + steps; ++step) {
    if (layer_.use_bias_) {
      reset_.row(step) += bias_reset_.W();
      update_.row(step) += bias_update_.W();
    }

    if (step != 0) {
      reset_.row(step).noalias()  += output_.row(step - 1) * syn_reset_out_.W().transpose();
      update_.row(step).noalias() += output_.row(step - 1) * syn_update_out_.W().transpose();
    }
    SigmoidActivation().Forward(reset_.row(step).data(), size_);
    SigmoidActivation().Forward(update_.row(step).data(), size_);

    if (step != 0) {
      partialhidden_.row(step).noalias() = output_.row(step - 1).cwiseProduct(reset_.row(step));
      quasihidden_.row(step).noalias() +=
          partialhidden_.row(step) * syn_quasihidden_out_.W().transpose();
    }
    TanhActivation().Forward(quasihidden_.row(step).data(), size_);

    if (step == 0) {
      output_.row(step).row(step).noalias()
          = quasihidden_.row(step).cwiseProduct(update_.row(step));
    } else {
      // these 3 lines means:
      // output_t = (quasihidden_t - output_{t - 1}) * update_t + output_{t - 1}
      output_.row(step).noalias() = quasihidden_.row(step) - output_.row(step - 1);
      output_.row(step).array() *= update_.row(step).array();
      output_.row(step) += output_.row(step - 1);
    }
  }
}

void GRULayer::Updater::BackwardSequence(
    int steps, uint32_t truncation_seed, int bptt_period, int bptt) {
  if (steps == 0) {
    return;
  }

  BackwardSequenceTruncated(steps, truncation_seed, bptt_period, bptt);

  input_g_.topRows(steps).setZero();
  if (layer_.use_input_weights_) {
    input_g_.topRows(steps).noalias() += quasihidden_g_.topRows(steps) * syn_update_in_.W();
    input_g_.topRows(steps).noalias() += reset_g_.topRows(steps) * syn_reset_in_.W();
    input_g_.topRows(steps).noalias() += update_g_.topRows(steps) * syn_update_in_.W();
  } else {
    input_g_.topRows(steps) += quasihidden_g_.topRows(steps);
    input_g_.topRows(steps) += reset_g_.topRows(steps);
    input_g_.topRows(steps) += update_g_.topRows(steps);
  }

  syn_quasihidden_out_.GetGradients()->noalias() =
      quasihidden_g_.middleRows(1, steps - 1).transpose()
      * partialhidden_.middleRows(1, steps - 1);

  syn_reset_out_.GetGradients()->noalias() =
      reset_g_.middleRows(1, steps - 1).transpose()
      * output_.topRows(steps - 1);

  syn_update_out_.GetGradients()->noalias() =
      update_g_.middleRows(1, steps - 1).transpose()
      * output_.topRows(steps - 1);

  if (layer_.use_input_weights_) {
    syn_quasihidden_in_.GetGradients()->noalias() =
      quasihidden_g_.middleRows(0, steps).transpose()
      * input_.topRows(steps);

    syn_reset_in_.GetGradients()->noalias() =
      reset_g_.middleRows(0, steps).transpose()
      * input_.topRows(steps);

    syn_update_in_.GetGradients()->noalias() =
      update_g_.middleRows(0, steps).transpose()
      * input_.topRows(steps);
  }

  if (layer_.use_bias_) {
    bias_reset_.GetGradients()->noalias() =
      reset_g_.topRows(steps).colwise().sum();
    bias_update_.GetGradients()->noalias() =
      update_g_.topRows(steps).colwise().sum();
  }
}

void GRULayer::Updater::BackwardStep(int step) {
  update_g_.row(step) = quasihidden_.row(step);
  if (step != 0) {
    update_g_.row(step) -= output_.row(step - 1);
  }
  update_g_.row(step).array() *= output_g_.row(step).array();
  SigmoidActivation().Backward(update_.row(step).data(), size_, update_g_.row(step).data());

  quasihidden_g_.row(step) = output_g_.row(step).cwiseProduct(update_.row(step));
  TanhActivation().Backward(quasihidden_.row(step).data(), size_, quasihidden_g_.row(step).data());

  partialhidden_g_.row(step).noalias() = quasihidden_g_.row(step) * syn_quasihidden_out_.W();

  if (step != 0) {
    reset_g_.row(step) = partialhidden_g_.row(step).cwiseProduct(output_.row(step - 1));
  }
  SigmoidActivation().Backward(reset_.row(step).data(), size_, reset_g_.row(step).data());
}


void GRULayer::Updater::BackwardStepThroughTime(int step) {
  // these 2 lines means: h'_{t - 1} += (1 - u_t) * h'_t
  output_g_.row(step - 1) += output_g_.row(step);
  output_g_.row(step - 1) -= update_.row(step).cwiseProduct(output_g_.row(step));

  output_g_.row(step - 1) += reset_.row(step).cwiseProduct(partialhidden_g_.row(step));
  output_g_.row(step - 1).noalias() += reset_g_.row(step) * syn_reset_out_.W();
  output_g_.row(step - 1).noalias() += update_g_.row(step) * syn_update_out_.W();
}


void GRULayer::Updater::UpdateWeights(
    int steps, Real lrate, Real l2reg, Real rmsprop, Real gradient_clipping) {
  if (steps <= 1 || size_ == 0) {
    return;
  }

  *syn_quasihidden_out_.GetGradients() /= steps;
  syn_quasihidden_out_.ApplyGradients(lrate, l2reg, rmsprop, gradient_clipping);

  *syn_reset_out_.GetGradients() /= steps;
  syn_reset_out_.ApplyGradients(lrate, l2reg, rmsprop, gradient_clipping);

  *syn_update_out_.GetGradients() /= steps;
  syn_update_out_.ApplyGradients(lrate, l2reg, rmsprop, gradient_clipping);

  if (layer_.use_input_weights_) {
    *syn_quasihidden_in_.GetGradients() /= steps + 1;
    syn_quasihidden_in_.ApplyGradients(lrate, l2reg, rmsprop, gradient_clipping);

    *syn_reset_in_.GetGradients() /= steps + 1;
    syn_reset_in_.ApplyGradients(lrate, l2reg, rmsprop, gradient_clipping);

    *syn_update_in_.GetGradients() /= steps + 1;
    syn_update_in_.ApplyGradients(lrate, l2reg, rmsprop, gradient_clipping);
  }

  if (layer_.use_bias_ && steps > 1) {
    *bias_reset_.GetGradients() /= steps - 1;
    bias_reset_.ApplyGradients(lrate, l2reg, rmsprop, gradient_clipping);

    *bias_update_.GetGradients() /= steps - 1;
    bias_update_.ApplyGradients(lrate, l2reg, rmsprop, gradient_clipping);
  }
}

std::vector<WeightMatrixUpdater<RowMatrix>*> GRULayer::Updater::GetMatrices() {
  std::vector<WeightMatrixUpdater<RowMatrix>*> v;
  v.push_back(&syn_reset_in_);
  v.push_back(&syn_reset_out_);
  v.push_back(&syn_update_in_);
  v.push_back(&syn_update_out_);
  v.push_back(&syn_quasihidden_in_);
  v.push_back(&syn_quasihidden_out_);
  return v;
}


std::vector<WeightMatrixUpdater<RowVector>*> GRULayer::Updater::GetVectors() {
  std::vector<WeightMatrixUpdater<RowVector>*> v;
  v.push_back(&bias_reset_);
  v.push_back(&bias_update_);
  return v;
}

IRecUpdater* GRULayer::CreateUpdater() {
  return new Updater(this);
}

