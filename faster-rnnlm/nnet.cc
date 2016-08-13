#include "faster-rnnlm/nnet.h"

#include <algorithm>
#include <cmath>

#include "faster-rnnlm/layers/interface.h"
#include "faster-rnnlm/nce.h"
#include "faster-rnnlm/util.h"
#include "faster-rnnlm/words.h"

namespace {
const uint64_t kVersionStepSize = 10000;
const int kCurrentVersion = 6;
const unsigned kMaxLayerTypeName = 64;  // maximum size of layer name type in bytes (including \0)
const std::string kDefaultLayerType = "sigmoid";
};  // unnamed namespace


static void ReadHeader(FILE* file, NNetConfig* cfg, int* version_ptr) {
  const char* error_message = "model header";
  int& version = *version_ptr;
  uint64_t quazi_layer_size;
  FreadAllOrDie(&quazi_layer_size, sizeof(int64_t), 1, file, error_message);
  version = quazi_layer_size / kVersionStepSize;
  if (version < 0 || version > kCurrentVersion) {
    fprintf(stderr, "Bad model version: %d\n", version);
    exit(1);
  }
  cfg->layer_size = quazi_layer_size % kVersionStepSize;

  FreadAllOrDie(&cfg->maxent_hash_size, sizeof(int64_t), 1, file, error_message);
  FreadAllOrDie(&cfg->maxent_order, sizeof(int), 1, file, error_message);

  cfg->nce_lnz = 9;  // magic value for default lnz in old versions
  if (version == 0) {
    cfg->use_nce = false;
  } else if (version == 1) {
    cfg->use_nce = true;
  } else {
    FreadAllOrDie(&cfg->use_nce, sizeof(bool), 1, file, error_message);
    FreadAllOrDie(&cfg->nce_lnz, sizeof(Real), 1, file, error_message);
  }

  cfg->reverse_sentence = false;
  if (version >= 3) {
    FreadAllOrDie(&cfg->reverse_sentence, sizeof(bool), 1, file, error_message);
  }

  cfg->layer_type = kDefaultLayerType;
  if (version >= 4) {
    char buffer[kMaxLayerTypeName];
    FreadAllOrDie(buffer, sizeof(char), kMaxLayerTypeName, file, error_message);
    cfg->layer_type = buffer;
  }

  cfg->layer_count = 1;
  if (version >= 5) {
    FreadAllOrDie(&cfg->layer_count, sizeof(int), 1, file, error_message);
  }

  cfg->hs_arity = 2;
  if (version >= 6) {
    FreadAllOrDie(&cfg->hs_arity, sizeof(int), 1, file, error_message);
  }
}


static NNetConfig ReadConfig(const std::string& model_file) {
  FILE *file = fopen(model_file.c_str(), "rb");
  NNetConfig cfg;
  int version;
  ReadHeader(file, &cfg, &version);
  fclose(file);
  return cfg;
}


NNet::NNet(const Vocabulary& vocab, const NNetConfig& cfg, bool use_cuda,
           bool use_cuda_memory_efficient)
    : cfg(cfg)
    , vocab(vocab)
    , rec_layer(NULL)
    , nce(NULL)
    , use_cuda(use_cuda)
    , use_cuda_memory_efficient(use_cuda_memory_efficient)
{
  Init();
}


NNet::NNet(const Vocabulary& vocab, const std::string& model_file, bool use_cuda,
           bool use_cuda_memory_efficient)
    : cfg(ReadConfig(model_file))
    , vocab(vocab)
    , rec_layer(NULL)
    , nce(NULL)
    , use_cuda(use_cuda)
    , use_cuda_memory_efficient(use_cuda_memory_efficient)
{
  Init();
  ReLoad(model_file);
}


NNet::~NNet() {
  delete rec_layer;
  if (cfg.use_nce) {
    delete nce;
  } else {
    delete softmax_layer;
  }
}


void NNet::Init() {
  if (cfg.layer_type.size() + 1 > kMaxLayerTypeName) {
    fprintf(stderr, "ERROR layer type name must be less then %d\n", kMaxLayerTypeName);
    exit(1);
  }

  fprintf(stderr,
      "Constructing RNN: layer_size=%" PRId64 ", layer_type=%s, layer_count=%d,"
      " maxent_hash_size=%" PRId64 ", maxent_order=%d, vocab_size=%d, use_nce=%d\n",
      cfg.layer_size, cfg.layer_type.c_str(), cfg.layer_count, cfg.maxent_hash_size,
      cfg.maxent_order, vocab.size(), static_cast<int>(cfg.use_nce));
  embeddings.resize(vocab.size(), cfg.layer_size);
  if (cfg.layer_size) {
    InitNormal(std::min(1. / std::sqrt(cfg.layer_size), 0.01), &embeddings);
  }

  rec_layer = CreateLayer(cfg.layer_type, cfg.layer_size, cfg.layer_count);
  if (rec_layer == NULL) {
    fprintf(stderr, "ERROR failid to create a recurrent layer of type '%s'",
        cfg.layer_type.c_str());
    exit(1);
  }

  maxent_layer.Init(cfg.maxent_hash_size);

  if (cfg.use_nce) {
    nce = new NCE(use_cuda, use_cuda_memory_efficient,
        cfg.nce_lnz, cfg.layer_size, vocab, cfg.maxent_hash_size);
  } else {
    softmax_layer = HSTree::CreateHuffmanTree(vocab, cfg.layer_size, cfg.hs_arity);
    // softmax_layer = HSTree::CreateRandomTree(vocab, cfg.layer_size, cfg.hs_arity, 0);
  }
}


void NNet::ApplyDiagonalInitialization(Real alpha) {
  rec_layer->GetWeights()->DiagonalInitialization(alpha);
}

void NNet::Save(const std::string& model_file) const {
  if (
      !cfg.use_nce && !cfg.reverse_sentence && cfg.hs_arity == 2 &&
      cfg.layer_type == kDefaultLayerType && cfg.layer_count == 1) {
    return SaveCompatible(model_file);
  }

  FILE* file = fopen(model_file.c_str(), "wb");

  uint64_t encoded_layer_size = cfg.layer_size + kVersionStepSize * kCurrentVersion;
  fwrite(&encoded_layer_size, sizeof(int64_t), 1, file);
  fwrite(&cfg.maxent_hash_size, sizeof(int64_t), 1, file);
  fwrite(&cfg.maxent_order, sizeof(int), 1, file);

  fwrite(&cfg.use_nce, sizeof(bool), 1, file);
  fwrite(&cfg.nce_lnz, sizeof(Real), 1, file);

  fwrite(&cfg.reverse_sentence, sizeof(bool), 1, file);
  {
    char buffer[kMaxLayerTypeName] = {0};
    strncpy(buffer, cfg.layer_type.c_str(), kMaxLayerTypeName - 1);
    fwrite(buffer, sizeof(char), kMaxLayerTypeName, file);
  }

  fwrite(&cfg.layer_count, sizeof(int), 1, file);
  fwrite(&cfg.hs_arity, sizeof(int), 1, file);

  ::Dump(embeddings, file);
  if (cfg.use_nce) {
    nce->Dump(file);
  } else {
    softmax_layer->Dump(file);
  }
  rec_layer->GetWeights()->Dump(file);

  maxent_layer.Dump(file);
  fclose(file);
}


void NNet::SaveCompatible(const std::string& model_file) const {
  FILE* file = fopen(model_file.c_str(), "wb");

  fwrite(&cfg.layer_size, sizeof(int64_t), 1, file);
  fwrite(&cfg.maxent_hash_size, sizeof(int64_t), 1, file);
  fwrite(&cfg.maxent_order, sizeof(int), 1, file);

  ::Dump(embeddings, file);
  softmax_layer->Dump(file);
  rec_layer->GetWeights()->Dump(file);

  maxent_layer.Dump(file);
  fclose(file);
}


void NNet::ReLoad(const std::string& model_file) {
  FILE *file = fopen(model_file.c_str(), "rb");

  NNetConfig read_cfg;
  int read_version;
  ReadHeader(file, &read_cfg, &read_version);

  ::Load(&embeddings, file);
  if (cfg.use_nce) {
    nce->Load(file);
  } else {
    softmax_layer->Load(file);
  }
  rec_layer->GetWeights()->Load(file);
  maxent_layer.Load(file);
  fclose(file);
}
