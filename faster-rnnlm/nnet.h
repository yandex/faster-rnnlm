#ifndef FASTER_RNNLM_NNET_H_
#define FASTER_RNNLM_NNET_H_
#include <inttypes.h>
#include <stdio.h>

#include <string>

#include "faster-rnnlm/hierarchical_softmax.h"
#include "faster-rnnlm/maxent.h"
#include "faster-rnnlm/settings.h"
#include "faster-rnnlm/util.h"

class NCE;
class Vocabulary;
class IRecLayer;


struct NNetConfig {
  int64_t layer_size;
  int layer_count;
  uint64_t maxent_hash_size;
  int maxent_order;

  bool use_nce;
  Real nce_lnz;

  bool reverse_sentence;

  int hs_arity;

  std::string layer_type;
};


struct NNet {
  NNet(const Vocabulary& vocab, const NNetConfig& cfg, bool use_cuda,
       bool use_cuda_memory_efficient);
  NNet(const Vocabulary& vocab, const std::string& model_file, bool use_cuda,
       bool use_cuda_memory_efficient);
  ~NNet();

  void ApplyDiagonalInitialization(Real);

  void Save(const std::string& model_file) const;

  // Reload weights from the file
  //
  // The config of the model in the file is expected to be identical to cfg
  // It is up to user to guarantee this
  void ReLoad(const std::string& model_file);

  const NNetConfig cfg;
  const Vocabulary& vocab;

  // embedding weigts
  RowMatrix embeddings;
  // recurrent weights
  IRecLayer* rec_layer;

  HSTree* softmax_layer;

  NCE* nce;

  MaxEnt maxent_layer;

  const bool use_cuda;
  const bool use_cuda_memory_efficient;

 private:
  void Init();
  void SaveCompatible(const std::string& model_file) const;
};

#endif  // FASTER_RNNLM_NNET_H_
