/*
 * Numerical gradient tests
 *
 * The file contains tests that compare analytical gradients with numerical ones
 * for critertions and hidden layers
 */
#include <stdio.h>
#include <stdlib.h>

#include <map>
#include <iostream>
#include <functional>
#include <memory>
#include <vector>

#include "faster-rnnlm/hierarchical_softmax.h"
#include "faster-rnnlm/maxent.h"
#include "faster-rnnlm/util.h"
#include "faster-rnnlm/layers/activation_functions.h"
#include "faster-rnnlm/layers/gru_layer.h"
#include "faster-rnnlm/layers/scrn_layer.h"
#include "faster-rnnlm/layers/simple_layer.h"

#ifdef USE_DOUBLES
static const Real ArgEps = 1e-4;
static const Real ToleranceEps = 1e-5;
#else
static const Real ArgEps = 1e-3;
static const Real ToleranceEps = 2e-3;
#endif


template<class Matrix>
bool AreGradientsCorrect(
    Matrix& params,
    std::function<Real (const Matrix&)> calc_cost,
    std::function<Matrix (const Matrix&)> calc_grads) {
  Matrix analytical_grads = calc_grads(params);

  for (int r = 0; r < params.rows(); ++r) {
    for (int c = 0; c < params.cols(); ++c) {
      double analytical_grad = analytical_grads(r, c);
      double orig_value = params(r, c);

      params(r, c) = orig_value + ArgEps;
      double cost_plus = calc_cost(params);
      params(r, c) = orig_value - ArgEps;
      double cost_minus = calc_cost(params);
      double numerical_grad = (cost_plus - cost_minus) / (2. * ArgEps);
      params(r, c) = orig_value;

      if (std::abs(analytical_grad - numerical_grad) > ToleranceEps
          || isnan(analytical_grad) || isnan(numerical_grad)) {
        fprintf(stderr, "%f %f (%f - %f)\n",
            params(0, 0), analytical_grads(0, 0), cost_plus, cost_minus);
        fprintf(stderr,
            "ERROR: numerical gradient differs from analytical"
            " at pos %d,%d: (numerical != analytical) %f != %f (tolerance: %f)\n",
            r, c, numerical_grad, analytical_grad, ToleranceEps);
        return false;
      }
    }
  }
  return true;
}


bool IsDerivativeCorrect(Real param, std::function<Real (Real)> calc_cost,
                         std::function<Real (Real)> calc_grads) {
  typedef Eigen::Matrix<Real, 1, 1> ScalarMatrix;
  ScalarMatrix param_as_matrix = ScalarMatrix::Constant(param);
  return AreGradientsCorrect<ScalarMatrix>(
      param_as_matrix,
      [&calc_cost] (const ScalarMatrix& in) { return calc_cost(in(0)); },
      [&calc_grads](const ScalarMatrix& in) { return ScalarMatrix::Constant(calc_grads(in(0))); }
  );
}


void TestHS(const Vocabulary& vocab, int hidden_size, MaxEnt* maxent) {
  HSTree* tree = HSTree::CreateHuffmanTree(vocab, hidden_size);

  std::vector<uint64_t> feature_hashes;
  int maxent_order = 0;
  if (maxent != NULL) {
    maxent_order = rand() % 5;
    for (int i = 0; i < maxent_order; ++i) {
      feature_hashes.push_back(rand() % (maxent->GetHashSize() - vocab.size()));
    }
  }
  int target_word = rand() % vocab.size();

  auto compute_cost =
      [target_word, tree, maxent_order, maxent, feature_hashes] (const RowVector& hidden) {
    RowVector hidden_grad(hidden);
    Real log10prob = tree->PropagateForwardAndBackward(
        true, target_word, feature_hashes.data(), maxent_order,
        0, 0, 0, 0, 10,
        hidden.data(), hidden_grad.data(), maxent);
    return log10prob * log(10);
  };

  auto compute_grads =
      [target_word, tree, maxent_order, maxent, feature_hashes] (const RowVector& hidden) {
    RowVector hidden_grad(hidden);
    hidden_grad.setZero();
    tree->PropagateForwardAndBackward(
        true, target_word, feature_hashes.data(), maxent_order,
        0, 0, 0, 0, 10,
        hidden.data(), hidden_grad.data(), maxent);
    return hidden_grad;
  };

  RowVector hidden(hidden_size);
  InitUniform(1., &hidden);
  bool ok = AreGradientsCorrect<RowVector>(hidden, compute_cost, compute_grads);
  if (!ok) {
    exit(1);
  }

  delete tree;
};


void TestActivation(IActivation* activation, int size) {
  auto compute_cost = [activation] (Real input) {
    activation->Forward(&input, 1);
    return input;
  };
  auto compute_grads = [&](Real input) {
    Real grad = 1;
    activation->Forward(&input, 1);
    activation->Backward(&input, 1, &grad);
    return grad;
  };

  RowVector x(size);
  InitNormal(10, &x);
  for (int i = 0; i < size; ++i) {
    bool ok = IsDerivativeCorrect(x[i], compute_cost, compute_grads);
    if (!ok) {
      exit(1);
    }
  }
}


class SimpleCriterion {
 public:
  explicit SimpleCriterion(int size) : vector(size) {
    InitUniform(1, &vector);
  }

  Real Forward(const Ref<const RowMatrix> hidden_states, int steps) const {
    if (steps == 0) {
      return 0;
    }

    return (hidden_states.topRows(steps).array().square().matrix() * vector.transpose()).mean();
  }

  RowMatrix Backward(const Ref<const RowMatrix> hidden_states, int steps) const {
    RowMatrix hidden_grads = hidden_states;
    hidden_grads.setZero();
    for (int i = 0; i < steps; ++i) {
      hidden_grads.row(i) = vector.cwiseProduct(hidden_states.row(i));
      hidden_grads.row(i) *= 2. / steps;
    }
    return hidden_grads;
  }
 protected:
  RowVector vector;
};


bool TestHiddenLayerOutputToInput(IRecLayer* layer, int size, int steps) {
  std::shared_ptr<IRecUpdater> updater(layer->CreateUpdater());
  SimpleCriterion crit(size);

  auto compute_cost = [&] (const RowMatrix& input) {
    updater->GetInputMatrix().topRows(input.rows()) = input;
    updater->ForwardSequence(input.rows());
    return crit.Forward(updater->GetOutputMatrix(), input.rows());
  };
  auto compute_grads = [&](const RowMatrix& input) {
    updater->GetInputMatrix().topRows(input.rows()) = input;
    updater->ForwardSequence(input.rows());
    updater->GetOutputGradMatrix() = crit.Backward(updater->GetOutputMatrix(), input.rows());
    updater->BackwardSequence(input.rows(), 0, 0, 0);
    return updater->GetInputGradMatrix().topRows(input.rows());
  };

  RowMatrix input(steps, size);
  InitNormal(1, &input);

  return AreGradientsCorrect<RowMatrix>(input, compute_cost, compute_grads);
}


template<class Matrix, class T1, class T2>
bool TestHiddenLayerSingleWeightGradients(
    IRecLayer* layer, IRecUpdater* updater, const SimpleCriterion& crit,
    Matrix weight, int weight_idx,
    T1 GetWeights, T2 GetGradients,
    int steps) {

  auto compute_cost =
      [layer, updater, steps, crit, GetWeights, GetGradients] (const Matrix& weight) {
    *GetWeights(layer) = weight;
    updater->ForwardSequence(steps);
    return crit.Forward(updater->GetOutputMatrix(), steps);
  };

  auto compute_grads =
      [layer, updater, steps, crit, GetWeights, GetGradients] (const Matrix& weight) {
    *GetWeights(layer) = weight;
    updater->ForwardSequence(steps);
    updater->GetOutputGradMatrix() = crit.Backward(updater->GetOutputMatrix(), steps);
    updater->BackwardSequence(steps, 0, 0, 0);
    return *GetGradients(updater);
  };

  return AreGradientsCorrect<Matrix>(weight, compute_cost, compute_grads);
}


bool TestHiddenLayerWeightGradients(IRecLayer* layer, int size, int steps) {
  std::shared_ptr<IRecUpdater> updater(layer->CreateUpdater());
  SimpleCriterion crit(size);


  RowMatrix input(steps, size);
  InitNormal(1, &input);
  updater->GetInputMatrix().topRows(input.rows()) = input;

  int matrix_weight_count = layer->GetWeights()->GetMatrices().size();
  for (int weight_idx = 0; weight_idx < matrix_weight_count; ++weight_idx) {
    fprintf(stderr, "\rTesting weight matrix %d of %d\t", weight_idx + 1, matrix_weight_count);
    const RowMatrix& initial_weights = *layer->GetWeights()->GetMatrices()[weight_idx];
    bool ok = TestHiddenLayerSingleWeightGradients(
        layer, updater.get(), crit, initial_weights, weight_idx,
        [weight_idx] (IRecLayer* layer) {
          return layer->GetWeights()->GetMatrices()[weight_idx]; },
        [weight_idx] (IRecUpdater* updater) {
        return updater->GetMatrices()[weight_idx]->GetGradients(); },
        steps
    );

    if (!ok) {
      fprintf(stderr, "\n");
      return false;
    }
  }

  int vector_weight_count = layer->GetWeights()->GetVectors().size();
  for (int weight_idx = 0; weight_idx < vector_weight_count; ++weight_idx) {
    fprintf(stderr, "\rTesting weight vector %d of %d\t", weight_idx + 1, vector_weight_count);
    RowVector initial_weights = *layer->GetWeights()->GetVectors()[weight_idx];
    bool ok = TestHiddenLayerSingleWeightGradients(
        layer, updater.get(), crit, initial_weights, weight_idx,
        [weight_idx] (IRecLayer* layer) {
          return layer->GetWeights()->GetVectors()[weight_idx]; },
        [weight_idx] (IRecUpdater* updater) {
          return updater->GetVectors()[weight_idx]->GetGradients(); },
        steps
    );

    if (!ok) {
      fprintf(stderr, "\n");
      return false;
    }
  }
  fprintf(stderr, "\r                                              \r");
  return true;
}


int main(int argc, const char *argv[]) {
  Vocabulary vocab;
  vocab.Load("testing.vocab");

  std::map<std::string, std::function<IActivation*()> > activation_factories;
  activation_factories["sigmoid"] = [] { return new SigmoidActivation(); };
  activation_factories["tanh"] = [] { return new TanhActivation(); };
  activation_factories["relu"] = [] { return new ReLUActivation(); };
  activation_factories["relu-trunc"] = [] { return new TruncatedReLUActivation(); };

  {
    srand(0);
    std::vector<int> sizes;
    sizes.push_back(1);
    sizes.push_back(7);
    sizes.push_back(131);
    for (size_t i = 0; i < sizes.size(); ++i) {
      printf("Testing HS without maxent with hidden_size = %d\n", sizes[i]);
      for (int round = 0; round < 10; ++round) {
        TestHS(vocab, sizes[i], NULL);
      }
    }

    MaxEnt maxent;
    uint64_t hash_size = 101234;
    maxent.Init(hash_size);
    for (uint64_t i = 0; i < hash_size; ++i) {
      Real tmp;
      InitUniform(.1, 1, &tmp);
      maxent.UpdateValue(i, 1., tmp, 0);
    }
    for (size_t i = 0; i < sizes.size(); ++i) {
      printf("Testing HS with maxent with hidden_size = %d\n", sizes[i]);
      for (int round = 0; round < 10; ++round) {
        TestHS(vocab, sizes[i], &maxent);
      }
    }
  }

  {
    srand(0);
    int size = 100;
    int steps = 10;
    SimpleCriterion crit(size);
    RowMatrix hidden_states(steps, size);
    InitUniform(1., &hidden_states);
    fprintf(stderr, "Testing SimpleCriterion\n");
    for (int i = 0; i < steps + 1; ++i) {
      auto compute_cost = [=] (const RowMatrix& states) { return crit.Forward(states, steps); };
      auto compute_grads = [=](const RowMatrix& states) { return crit.Backward(states, steps); };
      if (!AreGradientsCorrect<RowMatrix>(hidden_states, compute_cost, compute_grads)) {
        exit(1);
      }
    }
  }

  for (auto& pair : activation_factories) {
    srand(0);
    std::shared_ptr<IActivation> activation(pair.second());
    printf("Testing activation %s\n", pair.first.c_str());
    TestActivation(activation.get(), 1000);
  }

  std::vector<const char*> layer_names = {
    "sigmoid", "tanh", "scrnfast0", "scrn0", "scrn10", "gru", "gru-insyn", "gru-full"
  };

  for (auto hidden_type : layer_names) {
    for (int count = 1; count <= 2; ++count) {
      int size = 50;
      {
        fprintf(stderr,
            "Testing correctness of weight-updater correspondance for '%s (count=%d)'\n",
            hidden_type, count);
        std::unique_ptr<IRecLayer> layer(CreateLayer(hidden_type, size, count));
        if (layer.get() == NULL) {
          fprintf(stderr, "ERROR create to build the network\n");
          exit(1);
        }

        std::shared_ptr<IRecUpdater> updater(layer->CreateUpdater());

        auto weight_matrices = layer->GetWeights()->GetMatrices();
        auto update_matrices = updater->GetMatrices();
        if (weight_matrices.size() != update_matrices.size()) {
          fprintf(stderr,
              "ERROR Count of weight matrices doesn't match count of updaters: %d != %d\n",
              static_cast<int>(weight_matrices.size()), static_cast<int>(update_matrices.size()));
          exit(1);
        }

        auto weight_vectors = layer->GetWeights()->GetVectors();
        auto update_vectors = updater->GetVectors();
        if (weight_vectors.size() != update_vectors.size()) {
          fprintf(stderr,
              "ERROR Count of weight vectors doesn't match count of updaters: %d != %d\n",
              static_cast<int>(weight_vectors.size()), static_cast<int>(update_vectors.size()));
          exit(1);
        }
      }
      std::vector<int> steps_range = {1, 2, 3, 5, 10};
      for (int steps : steps_range) {
        fprintf(stderr,
            "Testing hidden: type=%s, count=%d, size=%d, steps=%d\n",
            hidden_type, count, size, steps);
        for (int seed = 0; seed < 20; ++seed) {
          srand(seed);
          std::shared_ptr<IRecLayer> layer(CreateLayer(hidden_type, size, count));
          if (!TestHiddenLayerOutputToInput(layer.get(), size, steps)) {
            exit(1);
          }
          if (!TestHiddenLayerWeightGradients(layer.get(), size, steps)) {
            exit(1);
          }
        }
      }
    }
  }

  return 0;
}
