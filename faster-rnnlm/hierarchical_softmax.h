#ifndef FASTER_RNNLM_HIERARCHICAL_SOFTMAX_H_
#define FASTER_RNNLM_HIERARCHICAL_SOFTMAX_H_
#include <math.h>
#include <stdio.h>

#include <vector>

#include "faster-rnnlm/settings.h"
#include "faster-rnnlm/util.h"
#include "faster-rnnlm/words.h"


class MaxEnt;

class HSTree {
 public:
  // Factory function to build k-ary Huffman tree_ using the word counts
  // Frequent words will have short unique k-nary codes
  static HSTree* CreateHuffmanTree(const Vocabulary&, int layer_size, int arity);

  static HSTree* CreateRandomTree(const Vocabulary&, int layer_size, int arity, uint64_t seed);

  ~HSTree();

  void Dump(FILE* fo) const;

  void Load(FILE* fo);

  // Make ME hash act like Bloom filter: if the weight is zero, it was probably
  // never touched by training and this (an higher) ngrams
  // should not be considered for the target_word
  //
  // returns truncated maxent_order
  int DetectEffectiveMaxentOrder(
      WordIndex target_word, const MaxEnt* maxent,
      const uint64_t* feature_hashes, int maxent_order) const;

  // Propagate softmax forward and backward
  // given maxent and hidden layers
  //
  // Updates
  //   tree_->weights_, hidden_grad, maxent
  //
  // Returns
  //   log10 probability of the words, if calculate_probability is true
  //   0, otherwise
  //
  // feature_hashes is an array of offsets for the target_word
  // feature_hashes must contain at least maxent_order elements
  Real PropagateForwardAndBackward(
      bool calculate_probability, WordIndex target_word,
      const uint64_t* feature_hashes, int maxent_order,
      Real lrate, Real maxent_lrate, Real l2reg, Real maxent_l2reg, Real gradient_clipping,
      const Real* hidden,
      Real* hidden_grad, MaxEnt* maxent);


  // Propagate softmax forward and calculate probability
  // given maxent and hidden layers
  //
  //
  // Returns
  //   log10 probability of the word
  //
  // feature_hashes is an array of offsets for the target_word
  // feature_hashes must contain at least maxent_order elements
  Real CalculateLog10Probability(
      WordIndex target_word,
      const uint64_t* feature_hashes, int maxent_order,
      bool dynamic_maxent_prunning,
      const Real* hidden, const MaxEnt* maxent) const;


  // Sample a word given maxent and hidden layers
  //
  //
  // Returns
  //   log10 probability of the word
  //   sampled word
  //
  // feature_hashes is an array of offsets for the target_word
  // feature_hashes must contain at least maxent_order elements
  void SampleWord(
      const uint64_t* feature_hashes, int maxent_order,
      const Real* hidden, const MaxEnt* maxent,
      Real* logprob, WordIndex* sampled_word) const;

  const int layer_size;
  const size_t syn_size;
  RowMatrix weights_;

  class Tree;
  Tree* tree_;

 protected:
  HSTree(int vocab_size, int layer_size, int arity, const std::vector<int>& children);

 private:
  HSTree(const HSTree&);
  HSTree& operator=(const HSTree&);
};

#endif  // FASTER_RNNLM_HIERARCHICAL_SOFTMAX_H_
