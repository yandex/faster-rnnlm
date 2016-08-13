#include "faster-rnnlm/nce.h"

#define _USE_MATH_DEFINES // for VC++
#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#ifndef NOCUDA
#include "faster-rnnlm/cuda_softmax.h"
#endif
#include "faster-rnnlm/hierarchical_softmax.h"
#include "faster-rnnlm/maxent.h"
#include "faster-rnnlm/util.h"
#include "faster-rnnlm/words.h"


NCE::NCE(
        bool use_cuda, bool use_cuda_memory_efficient, Real zln, int layer_size,
        const Vocabulary& vocab, uint64_t maxent_hash_size)
    : zln_(zln)
    , layer_size_(layer_size)
    , vocab_size_(vocab.size())
    , maxent_hash_size_(maxent_hash_size)
    , sm_embedding_(vocab.size(), layer_size)
#ifndef NOCUDA
    , cust_(0)
    , use_cuda_(use_cuda)
#endif
{
  fprintf(stderr, "Constructing NCE: layer_size=%d, maxent_hash_size=%zu, cuda=%d, ln(Z)=%f\n",
      layer_size_, maxent_hash_size_, static_cast<int>(use_cuda), zln_);

  InitUniform(0.05, &sm_embedding_);

#ifdef NOCUDA
  if (use_cuda) {
    fprintf(stderr, "NCE error: Compiled without CUDA support!\n");
    exit(1);
  }
#else
  if (use_cuda_) {
    cust_ = new CudaStorage;
    InitCudaStorage(cust_, layer_size_, vocab_size_, maxent_hash_size_, zln_, use_cuda_memory_efficient);
  }
#endif
}

NCE::~NCE() {
#ifndef NOCUDA
  if (cust_) {
    FreeCudaStorage(cust_);
    delete cust_;
  }
  cust_ = NULL;
#endif
}

namespace {
uint64_t get_maxent_index(uint64_t initial_index, WordIndex word_index) {
  return initial_index + word_index;
}
}  // unnamed namespace

int NCE::DetectEffectiveMaxentOrder(
    WordIndex target_word, const MaxEnt* maxent,
    const uint64_t* maxent_indices, size_t maxent_size) const {
  for (size_t order = 0; order < maxent_size; ++order) {
    uint64_t maxent_index = get_maxent_index(maxent_indices[order], target_word);
    if (maxent->IsNull(maxent_index))
      return order;
  }
  return maxent_size;
}

Real NCE::CalculateWordLnScore(
    const Ref<const RowVector> hidden, const MaxEnt* maxent,
    const uint64_t* maxent_indices, int maxent_indices_count,
    WordIndex word) const {

  // Suprisingly, explicit loop works faster then the Eigen-based expression
  // Real score_rnnlm = hidden.cross(sm_embedding_.row(word));
  Real score_rnnlm = 0;
  const Real* embedding = sm_embedding_.data() + word * layer_size_;
  for (int i = 0; i < layer_size_; i++) {
    score_rnnlm += hidden.data()[i] * embedding[i];
  }

  for (int i = 0; i < maxent_indices_count; i++) {
    uint64_t maxent_index = get_maxent_index(maxent_indices[i], word);
    score_rnnlm += maxent->GetValue(maxent_index);
  }

  return score_rnnlm - zln_;
}

void NCE::Updater::PropagateForwardAndBackward(
    const Ref<const RowVector> hidden, WordIndex target_word,
    const uint64_t* maxent_indices, size_t maxent_size,
    const NoiseSample& sample, Real lrate, Real l2reg,
    Real maxent_lrate, Real maxent_l2reg, Real gradient_clipping,
    Ref<RowVector> hidden_grad, MaxEnt* maxent) {

  Real ln_sample_size = log(sample.size);
  for (int idx = 0; idx < sample.size + 1; idx++) {
    bool is_target = (idx == sample.size);
    int label = is_target ? 1 : -1;

    WordIndex word = is_target ? target_word : sample.noise_words[idx];

    Real logp_rnnlm = nce_->CalculateWordLnScore(hidden, maxent, maxent_indices, maxent_size, word);
    Real logp_noise = is_target ? sample.target_ln_probability : sample.noise_ln_probabilities[idx];
    logp_noise += ln_sample_size;

    Real signal_noise_ratio = exp(logp_rnnlm - logp_noise);
    Real numerator = is_target ? 1. : signal_noise_ratio;
    Real denominator = 1. + signal_noise_ratio;

    // calculate derivatives with respect to hidden
    // exlicit update is faster than eigen-based
    Real grad = label * numerator / denominator;
    Real* embedding = nce_->sm_embedding_.data() + word * nce_->layer_size_;
    for (int i = 0; i < nce_->layer_size_; i++) {
      hidden_grad.data()[i] += grad * embedding[i];
      embedding[i] *= (1 - l2reg);
      embedding[i] += lrate * Clip(grad * hidden.data()[i], gradient_clipping);
    }

    // update maxent weights
    Real maxent_grad = Clip(grad, gradient_clipping);
    for (size_t i = 0; i < maxent_size; i++) {
      uint64_t maxent_index = get_maxent_index(maxent_indices[i], word);
      maxent->UpdateValue(maxent_index, maxent_lrate, maxent_grad, maxent_l2reg);
    }
  }
}

void NCE::UploadNetWeightsToCuda(const MaxEnt* maxent) {
  if (use_cuda_) {
#ifndef NOCUDA
    std::vector<Real> maxent_weights;
    maxent->DumpWeights(&maxent_weights);
    UploadNetWeights(cust_, sm_embedding_.data(), maxent_weights.data());
#endif
  }
}

void NCE::CalculateLog10ProbabilityBatch(
    const Ref<const RowMatrix> hidden_layers, const MaxEnt* maxent,
    const uint64_t* maxent_indices_all, const int* maxent_indices_count_all,
    const WordIndex* sentence, int sentence_length,
    const bool do_not_normalize,
    std::vector<Real>* logprob_per_pos) {

  logprob_per_pos->resize(sentence_length);

  if (use_cuda_) {
    if (do_not_normalize) {
      fprintf(stderr, "Unnormalized cuda mode is not supported\n");
      exit(1);
    }
#ifndef NOCUDA
    CalculateSoftMax(
      cust_, hidden_layers.data(), maxent_indices_all, maxent_indices_count_all,
      sentence_length, sentence, logprob_per_pos->data());
#endif
  } else {
    for (int target = 1; target <= sentence_length; ++target) {
      const Ref<const RowVector> hidden = hidden_layers.row(target - 1);

      const uint64_t* maxent_indices = maxent_indices_all + MAX_NGRAM_ORDER * (target - 1);
      int maxent_size = maxent_indices_count_all[target - 1];

      Real target_ln_score = CalculateWordLnScore(
          hidden, maxent, maxent_indices, maxent_size, sentence[target]);

      if (!do_not_normalize) {
        Real Z = 0;
        for (int word = 0; word < vocab_size_; ++word) {
          Z += exp(CalculateWordLnScore(
              hidden, maxent, maxent_indices, maxent_size, word));
        }
        target_ln_score -= log(Z);
      }

      // convert to ln -> log10
      (*logprob_per_pos)[target - 1] = target_ln_score / log(10);
    }
  }
}

void NCE::Dump(FILE* fo) const {
  ::Dump(sm_embedding_, fo);
}

void NCE::Load(FILE* fo) {
  ::Load(&sm_embedding_, fo);
}


UnigramNoiseGenerator::UnigramNoiseGenerator(
        const Vocabulary& vocab, Real noise_power, Real noise_min_cells)
    : noise_power_(noise_power)
    , noise_min_cells_(noise_min_cells)
{
  const size_t vocab_size = vocab.size();

  if (std::numeric_limits<WordIndex>::max() <= vocab_size) {
    // word indices cannot fit within int type
    fprintf(stderr, "ERROR Vocabularyulary size is too big for UnigramNoiseGenerator\n");
    exit(1);
  }

  if (noise_min_cells_ * vocab_size > kUnigramTableSize) {
    fprintf(stderr, "ERROR Cannot use %.2f cells for a vocabulary of size %zu given %u cells\n",
        noise_min_cells_, vocab_size, kUnigramTableSize);
    exit(1);
  }

  std::vector<double> quasi_freqs(vocab_size);
  for (size_t word = 0; word < vocab_size; word++) {
    quasi_freqs[word] = pow(vocab.GetWordFrequency(word), noise_power_);
  }
  // Sum frequencies in reverse order to improve precision
  double train_words_pow = std::accumulate(quasi_freqs.rbegin(), quasi_freqs.rend(), 0.);

  // If noise_min_cells_ > 0, then apply additive smoothing
  double min_quasi_freq = *std::min_element(quasi_freqs.begin(), quasi_freqs.end());
  double addon = (
      (noise_min_cells_ * train_words_pow - min_quasi_freq * kUnigramTableSize)
      / (kUnigramTableSize - noise_min_cells_ * vocab_size));
  if (addon > 0) {
    // If addon <= 0 then we already have at least noise_min_cells_ cells per word
    for (size_t word = 0; word < vocab_size; word++) {
      quasi_freqs[word] += addon;
    }
    train_words_pow = std::accumulate(quasi_freqs.rbegin(), quasi_freqs.rend(), 0.);
  }

  // For each word calculate probability emit the word
  std::vector<double> probabilities(vocab_size);
  ln_probabilities_.resize(vocab_size);
  double ln_train_words_pow = log(train_words_pow);
  for (size_t word = 0; word < vocab_size; word++) {
     probabilities[word] = quasi_freqs[word] / train_words_pow;
     ln_probabilities_[word] = log(quasi_freqs[word]) - ln_train_words_pow;
  }

  unigram_table_.resize(kUnigramTableSize);
  const double kDoubleUnigramTableSize = kUnigramTableSize;
  double d1 = probabilities[0];
  size_t min_cell_size = kUnigramTableSize;
  size_t word = 0;
  for (size_t cell = 0, cell_start = 0; cell < kUnigramTableSize; cell++) {
    unigram_table_[cell] = word;
    if (cell / kDoubleUnigramTableSize > d1 && word < vocab_size) {
      min_cell_size = std::min(min_cell_size, cell - cell_start + 1);
      cell_start = cell + 1;
      word++;
      d1 += probabilities[word];
    }
  }

  if (word != vocab_size - 1) {
    min_cell_size = 0;
  }

  fprintf(stderr,
      "Constructed UnigramNoiseGenerator:"
      " power=%.3f, mincells(param)=%.3f, mincells(real)=%zu\n",
      noise_power_, noise_min_cells_, min_cell_size);
}

uint64_t UnigramNoiseGenerator::PrepareNoiseSample(
    uint64_t random_state, int n_samples, const WordIndex* sen, int sen_pos,
    NoiseSample* sample) const {
  if (n_samples > kMaxNoiseSamples) {
    fprintf(stderr, "ERROR: Cannot use more then %d noise samples!\n", kMaxNoiseSamples);
    exit(1);
  }

  for (int i = 0; i < n_samples; ++i) {
    uint32_t random_value = GetNextRandom(&random_state);
    WordIndex word = unigram_table_[random_value % kUnigramTableSize];
    sample->noise_words[i] = word;
    sample->noise_ln_probabilities[i] = ln_probabilities_[word];
  }

  sample->size = n_samples;
  sample->target_ln_probability = ln_probabilities_[sen[sen_pos]];
  return random_state;
}


HSMaxEntNoiseGenerator::HSMaxEntNoiseGenerator(
        const HSTree* tree, const MaxEnt* maxent_layer, uint64_t maxent_hash_size,
        int vocab_size, int maxent_order)
    : tree_(tree)
    , maxent_layer_(maxent_layer)
    , maxent_hash_size_(maxent_hash_size)
    , vocab_size_(vocab_size)
    , maxent_order_(maxent_order)
{
  fprintf(stderr, "Constructed HSMaxEntNoiseGenerator\n");
}


uint64_t HSMaxEntNoiseGenerator::PrepareNoiseSample(
    uint64_t random_state, int n_samples, const WordIndex* sen, int sen_pos,
    NoiseSample* sample) const {
  if (n_samples > kMaxNoiseSamples) {
    fprintf(stderr, "ERROR: Cannot use more then %d negative samples!\n", kMaxNoiseSamples);
    exit(1);
  }

  uint64_t ngram_hashes[MAX_NGRAM_ORDER];
  const bool kMaxentAddPadding = false;
  int maxent_present = CalculateMaxentHashIndices(
      sen, sen_pos, maxent_order_, maxent_hash_size_ - vocab_size_,
      kMaxentAddPadding, ngram_hashes);

  for (int i = 0; i < n_samples; ++i) {
    tree_->SampleWord(
        ngram_hashes, maxent_present,
        NULL, maxent_layer_,
        &sample->noise_ln_probabilities[i], &sample->noise_words[i]);
  }

  for (int i = 0; i < n_samples; ++i) {
    // convert log10 to ln
    sample->noise_ln_probabilities[i] *= M_LN10;
  }

  const bool kMaxentPruning = false;
  sample->size = n_samples;
  sample->target_ln_probability = M_LN10 * tree_->CalculateLog10Probability(
      sen[sen_pos], ngram_hashes, maxent_present, kMaxentPruning,
      NULL, maxent_layer_);

  return random_state;
}
