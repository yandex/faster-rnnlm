#ifndef FASTER_RNNLM_CUDA_SOFTMAX_H_
#define FASTER_RNNLM_CUDA_SOFTMAX_H_

#include <inttypes.h>

#include <stddef.h>

#include "settings.h"

struct CudaStorageInner;

struct CudaStorage {
  Real* sm_embedding;
  Real* maxent;
  Real* hidden_layers;

  uint64_t* maxent_indices_all;
  int* maxent_indices_count_all;

  Real* scores;

  Real* vocab_ones;
  Real* sententence_Z;
  Real* sententence_Z_host;

  Real* target_scores;
  Real* target_scores_host;
  Real* logprobs;
  WordIndex* sen_shifted;
  size_t layer_size, vocab_size, maxent_hash_size;
  Real lnz;

  CudaStorageInner* inner;
};

extern "C"
void InitCudaStorage(CudaStorage* cust, size_t layer_size, size_t vocab_size, size_t maxent_hash_size, Real lnz);

extern "C"
void FreeCudaStorage(CudaStorage* cust);

extern "C"
void UploadNetWeights(CudaStorage* cust, const Real* sm_embedding_cpu, const Real* maxent);

extern "C"
void CalculateSoftMax(CudaStorage* cust, const Real* hidden_layers, const uint64_t* maxent_indices_all, const int* maxent_indices_count_all, size_t sentence_length, const WordIndex* sen, Real* logprobs);

#endif  // FASTER_RNNLM_CUDA_SOFTMAX_H_
