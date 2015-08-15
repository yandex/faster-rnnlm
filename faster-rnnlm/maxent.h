#ifndef FASTER_RNNLM_MAXENT_H_
#define FASTER_RNNLM_MAXENT_H_

#include <inttypes.h>
#include <stdio.h>

#include <cmath>
#include <vector>

#include "faster-rnnlm/settings.h"

// This list of primes is taken from Mikolov's RNNLM for the sake of backward compatibility
static const uint64_t PRIMES[] = {
    108641969, 116049371, 125925907, 133333309, 145678979, 175308587, 197530793, 234567803,
    251851741, 264197411, 330864029, 399999781, 407407183, 459258997, 479012069, 545678687,
    560493491, 607407037, 629629243, 656789717, 716048933, 718518067, 725925469, 733332871,
    753085943, 755555077, 782715551, 790122953, 812345159, 814814293, 893826581, 923456189,
    940740127, 953085797, 985184539, 990122807};
static const uint64_t PRIMES_SIZE = sizeof(PRIMES) / sizeof(PRIMES[0]);

// Calculate hashes of ngrams that preceeds word_index in the sentence
//
// Fills ngram_hashes array with numbers in {0, 1, ..., max_hash_index - 1}
// If add_padding then dummy -1 words are added to the beginning of the sentence
// Return the number of indices filled (maybe less than maxent_order
// for a few first sentence words if add_padding is false)
int CalculateMaxentHashIndices(
    const WordIndex *sen, int word_index,
    int maxent_order, uint64_t max_hash_index,
    bool add_padding,
    uint64_t ngram_hashes[MAX_NGRAM_ORDER]);


class MaxEnt {
 public:
  enum LearningMethod {kSGD, kAdaGrad, kFTRL};

  static const LearningMethod kLearningMethod = kSGD;
  static const size_t kStride = (kLearningMethod == kSGD) ? 1 : (kLearningMethod == kFTRL ? 3 : 2);

  MaxEnt() : hash_size_(0), storage_(NULL) {
    // empty constructor
  }

  void Init(size_t hash_size);

  size_t GetHashSize() const { return hash_size_; }

  ~MaxEnt();

  void DumpWeights(std::vector<Real>* weights) const;

  Real GetValue(size_t feature_index) const {
    return storage_[feature_index * kStride];
  }

  Real operator()(size_t feature_index) const {
    return GetValue(feature_index);
  }

  bool IsNull(size_t feature_index) const {
    return std::abs(GetValue(feature_index)) < 1e-6;
  }

  void UpdateValue(size_t feature_index, Real learning_rate, Real antigrad, Real l2) {
    if (kLearningMethod == kAdaGrad) {
      const Real kAdaGradPower = 0.5;

      Real& n = storage_[feature_index * kStride + 1];
      n += antigrad * antigrad;
      Real adarate = n;
      if (kAdaGradPower == 0.5) {
        adarate = std::sqrt(adarate);
      } else if (kAdaGradPower != 1) {
        adarate = std::pow(adarate, kAdaGradPower);
      }
      learning_rate /= adarate;
      Real& weight = storage_[feature_index * kStride];
      weight += learning_rate * antigrad - weight * l2;
    } else if (kLearningMethod == kFTRL) {
      // TODO: add a separate parameter for l1
      const Real l1 = l2;
      FTRLUpdate(feature_index, learning_rate, antigrad, l1, l2);
    } else {
      // SGD by default
      storage_[feature_index] += learning_rate * antigrad - storage_[feature_index] * l2;
    }
  }

  void Dump(FILE* fo) const;

  void Load(FILE* fo);

 private:
  size_t hash_size_;
  Real* storage_;

  void FTRLUpdate(size_t feature_index, Real learning_rate, Real antigrad, Real l1, Real l2) {
    // see http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf
    const Real alpha = learning_rate;
    const Real beta = 0;

    Real& w = storage_[feature_index * kStride];
    Real& n = storage_[feature_index * kStride + 1];
    Real& z = storage_[feature_index * kStride + 2];

    Real sigma = (std::sqrt(n + antigrad * antigrad) - std::sqrt(n)) / alpha;
    z -= antigrad + sigma * w;
    n += antigrad * antigrad;

    Real sign = (z < 0) ? -1 : 1;
    if (sign * z > l1) {
      w = (sign * l1 - z) / ((beta + std::sqrt(n)) / alpha + l2);
    } else {
      w = 0;
    }
  }
};


inline int CalculateMaxentHashIndices(
    const WordIndex *sen, int word_index,
    int maxent_order, uint64_t max_hash,
    bool add_padding,
    uint64_t ngram_hashes[MAX_NGRAM_ORDER]) {

  int maxent_present = (maxent_order > word_index + 1 && !add_padding) ? word_index + 1 : maxent_order;

  if (maxent_present) {
    // (order < maxent_present) <--> (order < maxent_order && order <= word_index)
    for (int order = 0; order < maxent_present; ++order) {
      ngram_hashes[order] = PRIMES[0] * PRIMES[1];
      for (int i = 1; i <= order; ++i) {
        uint64_t word = (word_index - i >= 0) ? sen[word_index - i] : -1;
        ngram_hashes[order] += PRIMES[(order * PRIMES[i] + i) % PRIMES_SIZE] * (word + 1);
      }
      ngram_hashes[order] = ngram_hashes[order] % max_hash;
    }
  }
  return maxent_present;
}

#endif  // FASTER_RNNLM_MAXENT_H_
