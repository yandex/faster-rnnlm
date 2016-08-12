#include "faster-rnnlm/hierarchical_softmax.h"

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "faster-rnnlm/maxent.h"
#include "faster-rnnlm/util.h"

using std::vector;

static const int kDynamic = 0;


class HSTree::Tree {
 public:
  // children array defines for each inner node defines its children
  Tree(int leaf_count, const vector<int>& children, int arity);

  int GetRootNode() const { return root_node_; }

  int GetTreeHeight() const { return tree_height_; }

  int GetArity() const { return arity_; }

  bool IsLeaf(int node) const { return (node < leaf_count_); }

  // get path_lengths for the word, i.e. the length of the path
  // from the root to the word
  int GetPathLength(WordIndex word) const { return path_lengths_[word]; }
  int& GetPathLength(WordIndex word) { return path_lengths_[word]; }

  // get array of path length node ids that lead from root to the word
  const int* GetPathToLeaf(WordIndex word) const {
    return points_.data() + word * (MAX_HSTREE_DEPTH + 1);
  }
  int* GetPathToLeaf(WordIndex word) {
    return points_.data() + word * (MAX_HSTREE_DEPTH + 1);
  }

  // get array of (path_lengths - 1) indices of branch ids that lead from root to the word
  const int* GetBranchPathToLead(WordIndex word) const {
    return branches_.data() + word * MAX_HSTREE_DEPTH;
  }
  int* GetBranchPathToLead(WordIndex word) {
    return branches_.data() + word * MAX_HSTREE_DEPTH;
  }

  // get array of size arity_ that contains indices of the node's children
  // 'node' must be inner node (node >= vocab_size)
  const int* GetChildren(int node) const {
    return children_.data() + (node - leaf_count_) * arity_;
  }

  // Get offset that corresponds to the child node in weights matrix
  //
  // Child node is the child number 'branch' of the node 'node_id'
  //
  // 'branch' belongs to {0, ..., arity_ - 2}
  // 'node' must be inner node (node >= vocab_size)
  int GetChildOffset(int node, int branch) const {
    return (node - leaf_count_) * (arity_ - 1) + branch;
  }

  // Get offset that corresponds to the child node in weights matrix
  //
  // Child node is the child number 'branch' of the node on depth 'depth'
  // on the path from the root to the word
  //
  // 'branch' belongs to {0, ..., arity_ - 2}
  int GetChildOffsetByDepth(WordIndex word, int depth, int branch) const {
    return GetChildOffset(GetPathToLeaf(word)[depth], branch);
  }

 protected:
  const int leaf_count_;
  const int arity_;
  int root_node_;
  int tree_height_;

  vector<int> children_;
  vector<int> path_lengths_;
  vector<int> points_;
  vector<int> branches_;
};


// Constructs a tree with leaf_count leaf nodes using a list of children
//
// It assumed that a complete tree of given could be constructed,
// i.e. (leaf_count - 1) must be divisible by (arity - 1)
//
// First leaf_count nodes are leaf nodes
// The next (leaf_count - 1) / (arity - 1) nodes are inner nodes
//
// children array could be considered as a row-based matrix
// with (inner nodes) rows and (arity) cols
// Element (i, j) is an index of j-th child of (leaf_count + i)-th node
HSTree::Tree::Tree(int leaf_count, const vector<int>& children, int arity)
  : leaf_count_(leaf_count)
  , arity_(arity)
  , root_node_(-1)     // to be filled
  , tree_height_(-1)   // to be filled
  , children_(children)
  , path_lengths_(leaf_count)
  , points_(leaf_count * (MAX_HSTREE_DEPTH + 1))
  , branches_(leaf_count * MAX_HSTREE_DEPTH)
{
  const int extra_node_count = (leaf_count - 1) / (arity_ - 1);
  const int node_count = leaf_count + extra_node_count;
  vector<int> branch_id(node_count);
  vector<int> parent_node(node_count);

  if ((leaf_count - 1) % (arity_ - 1) != 0) {
    fprintf(stderr,
        "Cannot build a full tree of arity %d from %d leaf nodes\n",
        arity_, leaf_count);
    exit(1);
  }

  root_node_ = node_count - 1;

  // build parents by children
  for (int parent = 0; parent < extra_node_count; parent++) {
    for (int branch = 0; branch < arity_; ++branch) {
      int child_index = children_[parent * arity_ + branch];
      if (child_index >= node_count) {
        fprintf(stderr, "ERROR bad child index (%d)\n", child_index);
        exit(1);
      }
      parent_node[child_index] = leaf_count + parent;
      branch_id[child_index] = branch;
    }
  }

  // Now assign branch_id code to each vocabulary word
  for (int leaf_node = 0; leaf_node < leaf_count; leaf_node++) {
    int path_nodes[MAX_HSTREE_DEPTH];
    int path_branches[MAX_HSTREE_DEPTH];
    int path_length = 0;
    for (int node = leaf_node; node != root_node_; node = parent_node[node]) {
      if (path_length == MAX_HSTREE_DEPTH) {
        fprintf(stderr, "ERROR Cannot build a tree with height greater than %d\n",
            MAX_HSTREE_DEPTH);
        exit(1);
      }
      path_branches[path_length] = branch_id[node];
      path_nodes[path_length] = node;
      path_length++;
    }

    GetPathLength(leaf_node) = path_length + 1;

    int* tree_path_nodes = GetPathToLeaf(leaf_node);
    int* tree_path_branches = GetBranchPathToLead(leaf_node);
    tree_path_nodes[0] = root_node_;
    for (int i = 0; i < path_length; i++) {
      tree_path_branches[path_length - i - 1] = path_branches[i];
      tree_path_nodes[path_length - i] = path_nodes[i];
    }
  }

  tree_height_ = 0;
  for (int i = 0; i < leaf_count_; i++) {
    tree_height_ = std::max(tree_height_, GetPathLength(i) - 1);
  }
}

// Finish construction of HSTree given constructed children array
HSTree::HSTree(int vocab_size, int layer_size, int arity, const vector<int>& children)
  : layer_size(layer_size)
  , syn_size(static_cast<size_t>(layer_size) * vocab_size)
  , weights_(vocab_size, layer_size)
  , tree_(new Tree(vocab_size, children, arity))
{
  if (layer_size) {
    InitNormal(1. / std::sqrt(layer_size), &weights_);
  }

  fprintf(stderr,
      "Contructed HS: arity=%d, height=%d\n",
      tree_->GetArity(), tree_->GetTreeHeight());
}

HSTree* HSTree::CreateHuffmanTree(const Vocabulary& vocab, int layer_size, int arity) {
  const int extra_node_count = (vocab.size() - 1) / (arity - 1);
  const int node_count = vocab.size() + extra_node_count;
  vector<int64_t> weight(node_count + 2);
  vector<int> children(extra_node_count * arity);

  for (int i = 0; i < vocab.size(); i++) {
    weight[i] = vocab.GetWordFrequency(i);
  }
  for (int i = vocab.size(); i < node_count; i++) {
    weight[i] = std::numeric_limits<int64_t>::max();
  }

  int next_leaf_node = vocab.size() - 1;
  int next_inner_node = vocab.size();

  vector<int> min_indices(arity);
  for (int new_node = vocab.size(); new_node < node_count; new_node++) {
    // First, find exactly arity smallest nodes
    // and store their indices in min_indices[new_node]
    for (int branch = 0; branch < arity; ++branch) {
      if (next_leaf_node >= 0 && weight[next_leaf_node] < weight[next_inner_node]) {
        min_indices[branch] = next_leaf_node;
        next_leaf_node--;
      } else {
        min_indices[branch] = next_inner_node;
        next_inner_node++;
      }
    }

    // Then, build a new node
    weight[new_node] = 0;
    for (int branch = 0; branch < arity; ++branch) {
      int child_index = min_indices[branch];
      weight[new_node] += weight[child_index];
      children[(new_node - vocab.size()) * arity + branch] = child_index;
    }
  }

  return new HSTree(vocab.size(), layer_size, arity, children);
}

HSTree* HSTree::CreateRandomTree(const Vocabulary& vocab, int layer_size, int arity, uint64_t seed) {
  const int extra_node_count = (vocab.size() - 1) / (arity - 1);
  vector<int> children(extra_node_count * arity);
  for (int i = 0; i < extra_node_count * arity; ++i) {
    children[i] = i;
  }
  if (extra_node_count > 0) {
    for (int i = 0; i < vocab.size(); i++) {
      int j = GetNextRandom(&seed) % (i + 1);
      std::swap(children[i], children[j]);
    }
  }
  return new HSTree(vocab.size(), layer_size, arity, children);
}

HSTree::~HSTree() {
  delete tree_;
}

void HSTree::Dump(FILE* fo) const {
  ::Dump(weights_, fo);
}

void HSTree::Load(FILE* fo) {
  ::Load(&weights_, fo);
}

// see the comment in the header
int HSTree::DetectEffectiveMaxentOrder(
    WordIndex target_word, const MaxEnt* maxent,
    const uint64_t* feature_hashes, int max_maxent_order) const {
  int maxent_order = max_maxent_order;
  for (int order = 0; order < max_maxent_order; ++order) {
    for (int d = 0; d < tree_->GetPathLength(target_word) - 1; d++) {
      for (int branch = 0; branch < tree_->GetArity() - 1; ++branch) {
        int child_offset = tree_->GetChildOffsetByDepth(target_word, d, branch);
        uint64_t maxent_index = feature_hashes[order] + child_offset;
        if (maxent->IsNull(maxent_index)) {
          return order;
        }
      }
    }
  }
  return maxent_order;
}

// Calculate inner product of the hidden layers with vectors
// that correspond to the children of the node
// except for the last one
//
// feature_hashes is an array of offsets for the target_word
// feature_hashes must contain at least maxent_order elements
template<int kArity>
inline void CalculateNodeChildrenScores(
    const HSTree* hs, int node, const Real* hidden,
    const uint64_t* feature_hashes, int maxent_order, const MaxEnt* maxent,
    Real* branch_scores) {
  const int arity = (kArity == kDynamic) ? hs->tree_->GetArity() : kArity;
  for (int branch = 0; branch < arity - 1; ++branch) {
    branch_scores[branch] = 0;
    int child_offset = hs->tree_->GetChildOffset(node, branch);
    const Real* sm_embedding = hs->weights_.row(child_offset).data();
    for (int i = 0; i < hs->layer_size; ++i) {
      branch_scores[branch] += hidden[i] * sm_embedding[i];
    }
    for (int order = 0; order < maxent_order; ++order) {
      uint64_t maxent_index = feature_hashes[order] + child_offset;
      branch_scores[branch] += maxent->GetValue(maxent_index);
    }
  }
}


// Emulate an array of fixed size
//
// If kStaticSize != kDynamic, then the array would be static,
// otherwise - dynamic
template<class T, int kStaticSize>
class MaybeStaticArray {
 public:
  explicit MaybeStaticArray(int dynamic_size)
    : dynamic_array((kStaticSize == kDynamic) ? new T[dynamic_size] : NULL) {}

  ~MaybeStaticArray() {
    if (kStaticSize == kDynamic) {
      delete dynamic_array;
    }
  }

  T* data() { return (kStaticSize == kDynamic) ? dynamic_array : static_array; }

  T& operator[](size_t idx) { return data()[idx]; }

 private:
  T* dynamic_array;
  T static_array[(kStaticSize == kDynamic) ? 0 : kStaticSize];
};


// Do a forward propagation for a softmax over children of the current node
// given hidden layer and maxent scores
//
// state must be an array of state_size elements
// feature_hashes is an array of offsets for the target_word
// feature_hashes must contain at least maxent_order elements
template<int kArity>
inline void PropagateNodeForward(
    const HSTree* hs, int node, const Real* hidden,
    const uint64_t* feature_hashes, int maxent_order, const MaxEnt* maxent,
    double* state) {
  const int arity = (kArity == kDynamic) ? hs->tree_->GetArity() : kArity;
  MaybeStaticArray<Real, kArity> tmp(arity);
  CalculateNodeChildrenScores<kArity>(hs, node, hidden, feature_hashes, maxent_order, maxent, tmp.data());

  double max_score = 0;
#ifdef PARANOID
  for (int i = 0; i < arity - 1; ++i) {
    max_score = (tmp[i] > max_score) ? tmp[i] : max_score;
  }
  state[arity - 1] = exp(-max_score);
#else
  state[arity - 1] = 1.;
#endif
  double f = state[arity - 1];
  for (int i = 0; i < arity - 1; ++i) {
    state[i] = exp(tmp[i] - max_score);
    f += state[i];
  }
  for (int i = 0; i < arity; ++i) {
    state[i] /= f;
  }
}


template<int kArity>
inline void PropagateNodeForwardByDepth(
    const HSTree* hs, WordIndex target_word, const Real* hidden, int depth,
    const uint64_t* feature_hashes, int maxent_order, const MaxEnt* maxent,
    double* state) {
  int node = hs->tree_->GetPathToLeaf(target_word)[depth];
  return PropagateNodeForward<kArity>(hs, node, hidden, feature_hashes, maxent_order, maxent, state);
}

// Do a back propagation of logloss critetion for each branch,
// the maxent layer, and the hidden layer
// given the softmax values that are stored in the state
//
// Updates
//   hs->weights_, hidden_grad, maxent
//
// state must be an array of state_size elements
// feature_hashes is an array of offsets for the target_word
// feature_hashes must contain at least maxent_order elements
template<int kArity>
inline void PropagateNodeBackward(
    HSTree* hs, WordIndex target_word, int depth,
    const uint64_t* feature_hashes, int maxent_order,
    Real lrate, Real maxent_lrate, Real l2reg, Real maxent_l2reg, Real gradient_clipping,
    const double* state,
    const Real* hidden,
    Real* hidden_grad, MaxEnt* maxent
    ) {
  // calculate gradients d(ln(sigma(x, selected_branch))) / d(x_i), i <= n - 1
  const int selected_branch = hs->tree_->GetBranchPathToLead(target_word)[depth];
  // gradient
  const int arity = (kArity == kDynamic) ? hs->tree_->GetArity() : kArity;
  const int kArityMinusOne = (kArity == kDynamic) ? kDynamic : (kArity - 1);
  MaybeStaticArray<Real, kArityMinusOne> branch_gradient(arity - 1);
  for (int branch = 0; branch < arity - 1; ++branch) {
    const int match = (branch == selected_branch);
    // gradient of logsoftmax
    branch_gradient[branch] = (match - static_cast<Real>(state[branch]));
  }

  for (int branch = 0; branch < arity - 1; ++branch) {
    Real grad = branch_gradient[branch];
    int child_offset = hs->tree_->GetChildOffsetByDepth(target_word, depth, branch);
    Real* sm_embedding = hs->weights_.row(child_offset).data();

    // Propagate errors output -> hidden
    for (int i = 0; i < hs->layer_size; ++i) {
      hidden_grad[i] += grad * sm_embedding[i];
    }

    // Learn weights hidden -> output
    for (int i = 0; i < hs->layer_size; ++i) {
      Real update = grad * hidden[i];
      sm_embedding[i] *= (1 - l2reg);
      sm_embedding[i] += lrate * Clip(update, gradient_clipping);
    }

    // update maxent weights
    Real maxent_grad = Clip(grad, gradient_clipping);
    for (int order = 0; order < maxent_order; ++order) {
      uint64_t maxent_index = feature_hashes[order] + child_offset;
      maxent->UpdateValue(maxent_index, maxent_lrate, maxent_grad, maxent_l2reg);
    }
  }
}

// This function is required to hide template machinery
//
// if kArity == kDynamic then _arity field of the Tree is used
// otherwised it is assumed that _arity is equal to kArity
template<int kArity>
Real PropagateForwardAndBackwardReal(
    HSTree* hs,
    bool calculate_probability, WordIndex target_word,
    const uint64_t* feature_hashes, int maxent_order,
    Real lrate, Real maxent_lrate, Real l2reg, Real maxent_l2reg, Real gradient_clipping,
    const Real* hidden,
    Real* hidden_grad, MaxEnt* maxent
    ) {
  MaybeStaticArray<double, kArity> softmax_state(hs->tree_->GetArity());
  Real logprob = 0.;
  for (int depth = 0; depth < hs->tree_->GetPathLength(target_word) - 1; depth++) {
    PropagateNodeForwardByDepth<kArity>(
        hs, target_word, hidden, depth,
        feature_hashes, maxent_order, maxent,
        softmax_state.data());

    PropagateNodeBackward<kArity>(
        hs, target_word, depth, feature_hashes, maxent_order,
        lrate, maxent_lrate, l2reg, maxent_l2reg, gradient_clipping,
        softmax_state.data(),
        hidden, hidden_grad, maxent);

    if (calculate_probability) {
      const int selected_branch = hs->tree_->GetBranchPathToLead(target_word)[depth];
      logprob += log10(softmax_state[selected_branch]);
    }
  }

  return logprob;
}

Real HSTree::PropagateForwardAndBackward(
    bool calculate_probability, WordIndex target_word,
    const uint64_t* feature_hashes, int maxent_order,
    Real lrate, Real maxent_lrate, Real l2reg, Real maxent_l2reg, Real gradient_clipping,
    const Real* hidden,
    Real* hidden_grad, MaxEnt* maxent
    ) {

  if (tree_->GetArity() == 2) {
    return PropagateForwardAndBackwardReal<2>(
        this, calculate_probability, target_word, feature_hashes, maxent_order,
        lrate, maxent_lrate, l2reg, maxent_l2reg, gradient_clipping,
        hidden, hidden_grad, maxent);
  }
  return PropagateForwardAndBackwardReal<kDynamic>(
      this, calculate_probability, target_word, feature_hashes, maxent_order,
      lrate, maxent_lrate, l2reg, maxent_l2reg, gradient_clipping,
      hidden, hidden_grad, maxent);
}

// see the comment in the header
Real HSTree::CalculateLog10Probability(
    WordIndex target_word,
    const uint64_t* feature_hashes, int maxent_order,
    bool dynamic_maxent_prunning,
    const Real* hidden, const MaxEnt* maxent) const {
  const int arity = tree_->GetArity();
  vector<double> softmax_state(arity);
  Real logprob = 0.;
  for (int depth = 0; depth < tree_->GetPathLength(target_word) - 1; depth++) {
    int node = tree_->GetPathToLeaf(target_word)[depth];

    if (dynamic_maxent_prunning) {
      for (int order = 0; order < maxent_order; ++order) {
        for (int branch = 0; branch < arity - 1; ++branch) {
          int child_offset = tree_->GetChildOffset(node, branch);
          uint64_t maxent_index = feature_hashes[order] + child_offset;
          if (maxent->IsNull(maxent_index)) {
            maxent_order = order;
            break;
          }
        }
      }
    }

    PropagateNodeForward<kDynamic>(
        this, node, hidden,
        feature_hashes, maxent_order, maxent,
        softmax_state.data());

    const int selected_branch = tree_->GetBranchPathToLead(target_word)[depth];
    logprob += log10(softmax_state[selected_branch]);
  }

  return logprob;
}


void HSTree::SampleWord(
    const uint64_t* feature_hashes, int maxent_order,
    const Real* hidden, const MaxEnt* maxent,
    Real* logprob_, WordIndex* sampled_word_) const {
  int node = tree_->GetRootNode();
  const int arity = tree_->GetArity();
  vector<double> probabilities(arity);

  Real logprob = 0.;
  while (!tree_->IsLeaf(node)) {
    // Propagate hidden -> output
    PropagateNodeForward<kDynamic>(
        this, node, hidden, feature_hashes, maxent_order, maxent,
        probabilities.data());

    double f = 0.;
    int selected_branch = arity - 1;
    Real random = rand() / (Real)RAND_MAX;
    for (int branch = 0; branch < arity - 1; ++branch) {
      f += probabilities[branch];
      if (f > random) {
        selected_branch = branch;
        break;
      }
    }
    logprob += log10(probabilities[selected_branch]);
    node = tree_->GetChildren(node)[selected_branch];
  }
  *logprob_ = logprob;
  *sampled_word_ = node;
}
