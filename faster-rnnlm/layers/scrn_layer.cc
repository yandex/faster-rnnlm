#include "faster-rnnlm/layers/scrn_layer.h"

#include "faster-rnnlm/layers/activation_functions.h"
#include "faster-rnnlm/layers/util.h"


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
  explicit Updater(SCRNLayer* layer)
      : IRecUpdater(layer->weights_->layer_size_)
      , layer_(*layer)
      , context_size_(layer_.weights_->context_size_)
      , hidden_size_(size_ - context_size_)
      , antidecay_context_(layer_.weights_->decay_context_)
      , context_(MAX_SENTENCE_WORDS, context_size_), context_g_(context_)
      , context_input_(context_), context_input_g_(context_)
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
        context_input_.middleRows(start, steps).noalias() =
          input_.middleRows(start, steps) * syn_context_.W().transpose();
      } else {
        context_input_.middleRows(start, steps) =
          input_.block(start, 0, steps, context_size_);
      }
      antidecay_context_.array() = 1.0 - decay_context_.W().array();
      context_.middleRows(start, steps) = context_input_.middleRows(start, steps);
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
    if (steps == 0) {
      return;
    }
    if (context_size_ > 0) {
      context_g_.topRows(steps) = output_g_.topLeftCorner(steps, context_size_);
    }
    hidden_g_.topRows(steps) = output_g_.topRightCorner(steps, hidden_size_);
    BackwardSequenceTruncated(steps, truncation_seed, bptt_period, bptt);

    // Compute input_g_
    // First propagate gradients context->input
    if (context_size_ > 0) {
      context_input_g_.topRows(steps).array() =
        context_g_.topRows(steps).array().rowwise() * antidecay_context_.array();

      if (layer_.use_input_weights_) {
        input_g_.topRows(steps).noalias() =
          context_input_g_.topRows(steps) * syn_context_.W();
      } else {
        input_g_.topLeftCorner(steps, context_size_) =
          context_input_g_.topRows(steps);
      }
    } else {
      input_g_.topRows(steps).setZero();
    }

    // Then add gradients hidden->input
    if (layer_.use_input_weights_) {
      input_g_.topRows(steps).noalias() += hidden_g_.topRows(steps) * syn_rec_input_.W();
    } else {
      input_g_.topRightCorner(steps, hidden_size_) = hidden_g_.topRightCorner(steps, hidden_size_);
    }

    syn_rec_hidden_.GetGradients()->noalias() =
      hidden_g_.middleRows(1, steps - 1).transpose() * hidden_.topRows(steps - 1);
    if (layer_.use_input_weights_) {
      syn_rec_input_.GetGradients()->noalias() =
        hidden_g_.topRows(steps).transpose() * input_.topRows(steps);
      syn_context_.GetGradients()->noalias() =
        context_input_g_.topRows(steps).transpose() * input_.topRows(steps);
    }
    if (context_size_ != 0) {
      syn_rec_context_.GetGradients()->noalias() =
        hidden_g_.topRows(steps).transpose() * context_.topRows(steps);

      decay_context_.GetGradients()->noalias() =
          context_g_.middleRows(1, steps - 1).cwiseProduct(
             context_.topRows(steps - 1) - context_input_.middleRows(1, steps - 1)
          ).colwise().sum();
      decay_context_.GetGradients()->array() +=
          context_g_.row(0).array() * (-1 * context_input_.row(0).array());
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

    *syn_rec_hidden_.GetGradients() /= steps;
    syn_rec_hidden_.ApplyGradients(lrate, l2reg, rmsprop, gradient_clipping);

    if (layer_.use_input_weights_) {
      *syn_rec_input_.GetGradients() /= steps + 1;
      syn_rec_input_.ApplyGradients(lrate, l2reg, rmsprop, gradient_clipping);
    }

    if (context_size_ != 0) {
      *syn_rec_context_.GetGradients() /= steps + 1;
      syn_rec_context_.ApplyGradients(lrate, l2reg, rmsprop, gradient_clipping);
    }
  }

  std::vector<WeightMatrixUpdater<RowMatrix>*> GetMatrices();
  std::vector<WeightMatrixUpdater<RowVector>*> GetVectors();

 private:
  SCRNLayer& layer_;
  const int context_size_;
  const int hidden_size_;
  RowVector antidecay_context_;
  RowMatrix context_, context_g_;
  RowMatrix context_input_, context_input_g_;
  RowMatrix hidden_, hidden_g_;

  WeightMatrixUpdater<RowMatrix> syn_context_;
  WeightMatrixUpdater<RowVector> decay_context_;

  WeightMatrixUpdater<RowMatrix> syn_rec_context_;
  WeightMatrixUpdater<RowMatrix> syn_rec_input_;
  WeightMatrixUpdater<RowMatrix> syn_rec_hidden_;
};


SCRNLayer::SCRNLayer(int layer_size, int context_size, bool use_input_weights)
  : weights_(new Weights(layer_size, context_size))
  , use_input_weights_(use_input_weights)
{
  // empty constructor
}

SCRNLayer::~SCRNLayer() { delete weights_; }

IRecUpdater* SCRNLayer::CreateUpdater() {
  return new Updater(this);
}

std::vector<WeightMatrixUpdater<RowMatrix>*> SCRNLayer::Updater::GetMatrices() {
  std::vector<WeightMatrixUpdater<RowMatrix>*> v;
  v.push_back(&syn_context_);
  v.push_back(&syn_rec_context_);
  v.push_back(&syn_rec_input_);
  v.push_back(&syn_rec_hidden_);
  return v;
}


std::vector<WeightMatrixUpdater<RowVector>*> SCRNLayer::Updater::GetVectors() {
  std::vector<WeightMatrixUpdater<RowVector>*> v;
  v.push_back(&decay_context_);
  return v;
}

IRecWeights* SCRNLayer::GetWeights() { return weights_; }

