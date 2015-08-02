#include "faster-rnnlm/recurrent.h"

#include <math.h>
#include <vector>


struct SigmoidActivation : public IActivation {
  void Forward(Real* hidden, int size) {
    for (int i = 0; i < size; i++) {
      hidden[i] = exp(hidden[i]) / (1 + exp(hidden[i]));
    }
  }

  void Backward(const Real* hidden, int size, Real* hidden_g) {
    for (int i = 0; i < size; ++i) {
      hidden_g[i] *= hidden[i] * (1 - hidden[i]);
    }
  }
};


struct TanhActivation : public IActivation {
  void Forward(Real* hidden, int size) {
    for (int i = 0; i < size; i++) {
      hidden[i] = tanh(hidden[i]);
    }
  }

  void Backward(const Real* hidden, int size, Real* hidden_g) {
    for (int i = 0; i < size; ++i) {
      hidden_g[i] *= (1 - hidden[i] * hidden[i]);
    }
  }
};


struct ReLUActivation : public IActivation {
  void Forward(Real* hidden, int size) {
    for (int i = 0; i < size; i++) {
      hidden[i] = (hidden[i] > 0) ? hidden[i] : 0;
    }
  }

  void Backward(const Real* hidden, int size, Real* hidden_g) {
    for (int i = 0; i < size; ++i) {
      hidden_g[i] *= static_cast<int>(hidden[i] > 0);
    }
  }
};


struct TruncatedReLUActivation : public IActivation {
  static const Real kTruncation = 20;

  void Forward(Real* hidden, int size) {
    for (int i = 0; i < size; i++) {
      hidden[i] = (hidden[i] > 0 && hidden[i] < kTruncation) ? hidden[i] : 0;
    }
  }

  void Backward(const Real* hidden, int size, Real* hidden_g) {
    for (int i = 0; i < size; ++i) {
      hidden_g[i] *= static_cast<int>(hidden[i] > 0 && hidden[i] < kTruncation);
    }
  }
};


// Helper function to update weights of a linear component within a recurrent layer
//
// Linear components looks like F_t(x) = A * x_t, where A is matrix and x is a vector
// per_step_values[t] contains value of x_t for timestamp t
// per_step_value_grads[t] contains value of dL/dF_t for timestamp t (L is a loss function)
//
// per_step_values and per_step_value_grads are assumed to have at least steps rows
//
// Matrix templates are required to handle cases, when per_step matrices are actually submatrices
template<class Matrix1, class Matrix2>
void CalculateRecurrentWeightGradients(
    int steps, const Matrix1& per_step_values, const Matrix2& per_step_value_grads,
    RowMatrix* weight_grads) {
  if (steps == 0) {
    weight_grads->setZero();
    return;
  }

  weight_grads->noalias() = (
      per_step_value_grads.topRows(steps).transpose()
      * per_step_values.topRows(steps));

  (*weight_grads) /= (steps + 1);
}

// Helper function to update weights of a linear component within a recurrent layer
//
// Linear components looks like F_t(x) = A * x_t, where A is matrix and x is a vector
// per_step_values[t] contains value of x_t for timestamp t
// per_step_value_grads[t] contains value of dL/dF_t for timestamp t (L is a loss function)
//
// per_step_values and per_step_value_grads are assumed to have at least steps rows
//
// weight_grads is a matrix that is used to accumulate gradients of dL/dA
// weight_mean_grads is a matrix that accumulates weight grads to apply rmsprop
// weights is the weight matrix, i.e. 'A'

// Matrix templates are required to handle cases, when per_step matrices are actually submatrices
template<class Matrix1, class Matrix2, class Matrix3>
void UpdateRecurrentSynWeights(
    int steps, Real lrate, Real l2reg, Real rmsprop, Real gradient_clipping,
    const Matrix1& per_step_values, const Matrix2& per_step_value_grads,
    WeightMatrixUpdater<Matrix3>* updater) {
  if (steps == 0) {
    return;
  }

  updater->GetGradients()->noalias() = (
      per_step_value_grads.topRows(steps).transpose()
      * per_step_values.topRows(steps));

  (*updater->GetGradients()) /= (steps + 1);

  updater->ApplyGradients(lrate, l2reg, rmsprop, gradient_clipping);
}


// Curiously recurring template pattern
template<class TUpdater>
class TruncatedBPTTMixin {
 protected:
  // Truncated Backpropagation in time
  //
  // truncation_seed - seed that determines BPTT unfoliding start
  // bptt_period - period of BPTT
  // bptt - number of steps for real through time propagation;
  //   if bptt = 0, no truncation is performed
  //   otherwise bptt must be less or equal to bptt_period
  void BackwardSequenceTruncated(int steps, uint32_t truncation_seed, int bptt_period, int bptt) {
    if (bptt == 0) {
      bptt_period = 1;
    }
    int bptt_phase = (truncation_seed + bptt_period - 1) % bptt_period;
    for (int step = steps - 1; step > 0; --step, --bptt_phase) {
      static_cast<TUpdater*>(this)->BackwardStep(step);
      if (bptt == 0 || (bptt_phase >= (bptt_period - bptt))) {
        static_cast<TUpdater*>(this)->BackwardStepThroughTime(step);
      }
      if (bptt_phase == 0) {
        bptt_phase = bptt_period;
      }
    }
    if (steps != 0) {
      static_cast<TUpdater*>(this)->BackwardStep(0);
    }
  }

  // Back prapogate everything except for hiden->hidden
  // Must be defined in derived class
  void BackwardStep(int step) { exit(1); }

  // Back propagate hidden_{step - 1} -> hidden_{step} (step > 0)
  // Must be defined in derived class
  void BackwardStepThroughTime(int step) { exit(1); }
};


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


class SimpleRecurrentLayer::Updater : public IRecUpdater, public TruncatedBPTTMixin<SimpleRecurrentLayer::Updater> {
 public:
  Updater(SimpleRecurrentLayer* layer)
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

void SimpleRecurrentLayer::Updater::BackwardSequence(int steps, uint32_t truncation_seed, int bptt_period, int bptt) {
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


void SimpleRecurrentLayer::Updater::UpdateWeights(int steps, Real lrate, Real l2reg, Real rmsprop, Real gradient_clipping) {
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


class GRULayer::Updater : public IRecUpdater, public TruncatedBPTTMixin<GRULayer::Updater> {
 public:
  Updater(GRULayer* layer)
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
      quasihidden_.row(step).noalias() += partialhidden_.row(step) * syn_quasihidden_out_.W().transpose();
    }
    TanhActivation().Forward(quasihidden_.row(step).data(), size_);

    if (step == 0) {
      output_.row(step).row(step).noalias() = quasihidden_.row(step).cwiseProduct(update_.row(step));
    } else {
      // these 3 lines means: output_t = (quasihidden_t - output_{t - 1}) * update_t + output_{t - 1}
      output_.row(step).noalias() = quasihidden_.row(step) - output_.row(step - 1);
      output_.row(step).array() *= update_.row(step).array();
      output_.row(step) += output_.row(step - 1);
    }
  }
}

void GRULayer::Updater::BackwardSequence(int steps, uint32_t truncation_seed, int bptt_period, int bptt) {
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


void GRULayer::Updater::UpdateWeights(int steps, Real lrate, Real l2reg, Real rmsprop, Real gradient_clipping) {
  if (steps <= 1 || size_ == 0) {
    return;
  }

  UpdateRecurrentSynWeights(steps - 1, lrate, l2reg, rmsprop, gradient_clipping,
      partialhidden_.middleRows(1, steps - 1), quasihidden_g_.middleRows(1, steps - 1),
      &syn_quasihidden_out_);
  UpdateRecurrentSynWeights(steps - 1, lrate, l2reg, rmsprop, gradient_clipping,
      output_, reset_g_.bottomRows(reset_g_.rows() - 1),
      &syn_reset_out_);
  UpdateRecurrentSynWeights(steps - 1, lrate, l2reg, rmsprop, gradient_clipping,
      output_, update_g_.bottomRows(update_g_.rows() - 1),
      &syn_update_out_);

  if (layer_.use_input_weights_) {
    UpdateRecurrentSynWeights(steps, lrate, l2reg, rmsprop, gradient_clipping,
        input_, quasihidden_g_,
        &syn_quasihidden_in_);
    UpdateRecurrentSynWeights(steps, lrate, l2reg, rmsprop, gradient_clipping,
        input_, reset_g_,
        &syn_reset_in_);
    UpdateRecurrentSynWeights(steps, lrate, l2reg, rmsprop, gradient_clipping,
        input_, update_g_,
        &syn_update_in_);
  }

  if (layer_.use_bias_) {
    *bias_reset_.GetGradients() = reset_g_.middleRows(1, steps - 1).colwise().mean();
    bias_reset_.ApplyGradients(lrate, l2reg, rmsprop, gradient_clipping);

    *bias_update_.GetGradients() = update_g_.middleRows(1, steps - 1).colwise().mean();
    bias_update_.ApplyGradients(lrate, l2reg, rmsprop, gradient_clipping);
  }
}


IRecUpdater* GRULayer::CreateUpdater() {
  return new Updater(this);
}


class SCRNLayer::Weights : public IRecWeights {
 public:
  Weights(int layer_size, int context_size)
      : IRecWeights(4, 1)
      , layer_size_(layer_size)
      , context_size_(context_size)

      , syn_context_(matrices_[0])
      , decay_context_(vectors_[0])
      , syn_rec_context_(matrices_[1])
      , syn_rec_input_(matrices_[2])
      , syn_rec_hidden_(matrices_[3])
  {
    syn_context_.resize(context_size, layer_size);
    decay_context_.resize(context_size);
    syn_rec_context_.resize(layer_size - context_size, context_size);
    syn_rec_input_.resize(layer_size - context_size, layer_size);
    syn_rec_hidden_.resize(layer_size - context_size, layer_size - context_size);

    for (size_t i = 0; i < matrices_.size(); ++i) {
      int n = matrices_[i].cols() + matrices_[i].rows();
      if (n) {
        InitNormal(1. / std::sqrt(n / 2.0), &matrices_[i]);
      }
    }
    decay_context_.setConstant(0.95);
  }

  void DiagonalInitialization(Real alpha) {}

 protected:
  const int layer_size_;
  const int context_size_;

  RowMatrix& syn_context_;
  RowVector& decay_context_;

  RowMatrix& syn_rec_context_;
  RowMatrix& syn_rec_input_;
  RowMatrix& syn_rec_hidden_;

  friend class SCRNLayer::Updater;
};


class SCRNLayer::Updater : public IRecUpdater, public TruncatedBPTTMixin<SCRNLayer::Updater> {
 public:
  Updater(SCRNLayer* layer)
      : IRecUpdater(layer->weights_->layer_size_)
      , layer_(*layer)
      , context_size_(layer_.weights_->context_size_)
      , hidden_size_(size_ - context_size_)
      , antidecay_context_(layer_.weights_->decay_context_)
      , context_(MAX_SENTENCE_WORDS, context_size_), context_g_(context_)
      , hidden_(MAX_SENTENCE_WORDS, size_ - context_size_), hidden_g_(hidden_)
      , syn_context_(&layer_.weights_->syn_context_)
      , decay_context_(&layer_.weights_->decay_context_)
      , syn_rec_context_(&layer_.weights_->syn_rec_context_)
      , syn_rec_input_(&layer_.weights_->syn_rec_input_)
      , syn_rec_hidden_(&layer_.weights_->syn_rec_hidden_)
  {
    // empty constructor
  }

  void ForwardSubSequence(int start, int steps) {
    if (context_size_ > 0) {
      if (layer_.use_input_weights_) {
        context_.middleRows(start, steps).noalias() =
          input_.middleRows(start, steps) * syn_context_.W().transpose();
      } else {
        context_.middleRows(start, steps) =
          input_.block(start, 0, steps, context_size_);
      }
      antidecay_context_.array() = 1.0 - decay_context_.W().array();
      context_.middleRows(start, steps).array().rowwise() *= antidecay_context_.array();
    }

    if (layer_.use_input_weights_) {
      hidden_.middleRows(start, steps).noalias() =
        input_.middleRows(start, steps) * syn_rec_input_.W().transpose();
    } else {
      hidden_.middleRows(start, steps) =
        input_.block(start, context_size_, steps, hidden_size_);
    }

    for (int step = start; step < start + steps; ++step) {
      if (step != 0) {
        hidden_.row(step).noalias() +=
          hidden_.row(step - 1) * syn_rec_hidden_.W().transpose();
      }
      if (context_size_ > 0) {
        if (step != 0) {
          context_.row(step).array() +=
            decay_context_.W().array() * context_.row(step - 1).array();
        }
        hidden_.row(step).noalias() +=
          context_.row(step) * syn_rec_context_.W().transpose();
      }
      SigmoidActivation().Forward(hidden_.row(step).data(), hidden_.cols());
    }

    if (context_size_ > 0) {
      output_.block(start, 0, steps, context_size_) = context_.middleRows(start, steps);
    }
    output_.block(start, context_size_, steps, hidden_size_) =
     hidden_.middleRows(start, steps);
  }

  void BackwardSequence(int steps, uint32_t truncation_seed, int bptt_period, int bptt) {
    if (context_size_ > 0) {
      context_g_.topRows(steps) = output_g_.topLeftCorner(steps, context_size_);
    }
    hidden_g_.topRows(steps) = output_g_.topRightCorner(steps, hidden_size_);
    BackwardSequenceTruncated(steps, truncation_seed, bptt_period, bptt);

    if (context_size_ > 0) {
      if (layer_.use_input_weights_) {
        input_g_.topRows(steps).noalias() = (
            context_g_.topRows(steps).array().rowwise() * antidecay_context_.array()
          ).matrix() * syn_context_.W();
      } else {
        input_g_.topLeftCorner(steps, context_size_) = (
            context_g_.topRows(steps).array().rowwise() * antidecay_context_.array()
          ).matrix();
      }
    } else {
      input_g_.topRows(steps).setZero();
    }

    if (layer_.use_input_weights_) {
      input_g_.topRows(steps).noalias() += hidden_g_.topRows(steps) * syn_rec_input_.W();
    } else {
      input_g_.topRightCorner(steps, hidden_size_) = hidden_g_.topRightCorner(steps, hidden_size_);
    }
  }

  void BackwardStep(int step) {
    SigmoidActivation().Backward(
      hidden_.row(step).data(), hidden_.cols(), hidden_g_.row(step).data());
    if (context_size_ > 0) {
      context_g_.row(step).noalias() += hidden_g_.row(step) * syn_rec_context_.W();
    }
  }

  void BackwardStepThroughTime(int step) {
    if (context_size_ > 0) {
      context_g_.row(step - 1).array() +=
        decay_context_.W().array() * context_g_.row(step).array();
    }

    hidden_g_.row(step - 1).noalias() +=
      hidden_g_.row(step) * syn_rec_hidden_.W();
  }

  void UpdateWeights(int steps, Real lrate, Real l2reg, Real rmsprop, Real gradient_clipping) {
    if (steps <= 1 || size_ == 0) {
      return;
    }
    UpdateRecurrentSynWeights(steps - 1, lrate, l2reg, rmsprop, gradient_clipping,
        hidden_, hidden_g_.middleRows(1, steps - 1),
        &syn_rec_hidden_);

    UpdateRecurrentSynWeights(steps, lrate, l2reg, rmsprop, gradient_clipping,
        input_, hidden_g_,
        &syn_rec_input_);

    if (context_size_ != 0) {
      UpdateRecurrentSynWeights(steps, lrate, l2reg, rmsprop, gradient_clipping,
          context_, hidden_g_,
          &syn_rec_context_);
    }
  }

 private:
  SCRNLayer& layer_;
  const int context_size_;
  const int hidden_size_;
  RowVector antidecay_context_;
  RowMatrix context_, context_g_;
  RowMatrix hidden_, hidden_g_;

  WeightMatrixUpdater<RowMatrix> syn_context_;
  WeightMatrixUpdater<RowVector> decay_context_;

  WeightMatrixUpdater<RowMatrix> syn_rec_context_;
  WeightMatrixUpdater<RowMatrix> syn_rec_input_;
  WeightMatrixUpdater<RowMatrix> syn_rec_hidden_;
};


SCRNLayer::SCRNLayer(int layer_size, int context_size, bool use_input_weights)
  : weights_(new Weights(layer_size, context_size))
  , context_size_(context_size)
  , use_input_weights_(use_input_weights)
{
  // empty constructor
}

SCRNLayer::~SCRNLayer() { delete weights_; }

IRecUpdater* SCRNLayer::CreateUpdater() {
  return new Updater(this);
}

IRecWeights* SCRNLayer::GetWeights() { return weights_; }

struct LayerStack : public IRecLayer {
  struct Updater : public IRecUpdater {
    Updater(LayerStack* stack) : IRecUpdater(stack->layer_size) {
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

    virtual void UpdateWeights(int steps, Real lrate, Real l2reg, Real rmsprop, Real gradient_clipping) {
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

    std::vector<IRecUpdater*> updaters;
  };

  struct Weights : public IRecWeights {
    explicit Weights(LayerStack* stack) : IRecWeights(0, 0), stack(stack) {}

    void Dump(FILE* fo) const {
      for (size_t i = 0; i < stack->layers.size(); ++i) {
        stack->layers[i]->GetWeights()->Dump(fo);
      }
    }

    void Load(FILE* fo) {
      for (size_t i = 0; i < stack->layers.size(); ++i) {
        stack->layers[i]->GetWeights()->Load(fo);
      }
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


IRecLayer* CreateSingleLayer(const std::string& layer_type, int layer_size, bool first_layer) {
  // SimpleRecurrentLayer arguments: (int size, bool use_input_weights, IActivation*)
  if (layer_type == "sigmoid") return new SimpleRecurrentLayer(layer_size, !first_layer, new SigmoidActivation());
  if (layer_type == "tanh") return new SimpleRecurrentLayer(layer_size, !first_layer, new TanhActivation());
  if (layer_type == "relu") return new SimpleRecurrentLayer(layer_size, !first_layer, new ReLUActivation());
  if (layer_type == "relu-trunc") return new SimpleRecurrentLayer(layer_size, !first_layer, new TruncatedReLUActivation());
  // GRULayer arguments: (int size, bool use_bias, bool use_input_weights)
  if (layer_type == "gru") return new GRULayer(layer_size, false, false);
  if (layer_type == "gru-bias") return new GRULayer(layer_size, true, false);
  if (layer_type == "gru-insyn") return new GRULayer(layer_size, false, true);
  if (layer_type == "gru-full") return new GRULayer(layer_size, true, true);
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
