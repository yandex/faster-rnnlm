#include "faster-rnnlm/maxent.h"

#include <stdlib.h>
#include <string.h>

#include "faster-rnnlm/util.h"


void InitArray(Real** array, size_t size, Real value) {
  free(*array);
  if (posix_memalign(reinterpret_cast<void **>(array), 128, size * sizeof(Real)) != 0) {
    fprintf(stderr, "ERROR: Failed to allocate memory for maxent layer\n");
    exit(1);
  }
  for (size_t i = 0; i < size; ++i) {
    (*array)[i] = value;
  }
}


void MaxEnt::Init(size_t hash_size) {
  hash_size_ = hash_size;
  InitArray(&storage_, hash_size_ * kStride, 0);

  if (kLearningMethod == kAdaGrad || kLearningMethod == kFTRL) {
    for (size_t i = 0; i < hash_size_; ++i) {
      storage_[i * kStride + 1] = 1;
    }
  }
}

MaxEnt::~MaxEnt() {
  free(storage_);
}

void MaxEnt::DumpWeights(std::vector<Real>* weights) const {
  weights->resize(hash_size_);
  for (size_t i = 0; i < hash_size_; ++i) {
    (*weights)[i] = storage_[i * kStride];
  }
}

void MaxEnt::Dump(FILE* fo) const {
  if (hash_size_ > 0) {
    if (kLearningMethod == kAdaGrad) {
      // for models learned with AdaGrad save only weights
      // as a result adagrad models a back-compatible with sgd models
      std::vector<Real> weights;
      DumpWeights(&weights);
      fwrite(weights.data(), sizeof(Real), hash_size_, fo);
    } else {
      fwrite(storage_, sizeof(Real), hash_size_ * kStride, fo);
    }
  }
}

void MaxEnt::Load(FILE* fo) {
  if (hash_size_ > 0) {
    if (kLearningMethod == kAdaGrad) {
      // when AdaGrad model is read, reset gradient sum to one
      FreadAllOrDie(storage_, sizeof(Real), hash_size_, fo, "maxent hash");
      for (size_t i = hash_size_; i-->0; ) {
        storage_[i * kStride] = storage_[i];
        storage_[i * kStride + 1] = 1;
      }
    } else {
      FreadAllOrDie(storage_, sizeof(Real), hash_size_ * kStride, fo, "maxent hash");
    }
  }
}
