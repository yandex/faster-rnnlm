#include <fenv.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifndef NOTHREAD
#include <pthread.h>
#endif
#ifdef NORT
#include <sys/time.h>
#endif

#include <string>
#include <vector>

#include "faster-rnnlm/hierarchical_softmax.h"
#include "faster-rnnlm/layers/interface.h"
#include "faster-rnnlm/nce.h"
#include "faster-rnnlm/nnet.h"
#include "faster-rnnlm/program_options.h"
#include "faster-rnnlm/settings.h"
#include "faster-rnnlm/util.h"
#include "faster-rnnlm/words.h"

namespace {

enum OOVPolicy {kSkipSentence, kConvertToUnk};

// Compile time parameters
const bool kMaxentAddPadding = false;
const bool kHSMaxentPrunning = true;
#ifdef NOCUDA
const bool kHaveCudaSupport = false;
#else
const bool kHaveCudaSupport = true;
#endif
const int64_t kReportEveryWords = 50000;
const OOVPolicy kOOVPolicy = kConvertToUnk;

// Run time learning parameters
// Constant outside main
// - common
Real initial_lrate = 0.1, initial_maxent_lrate = 0.1;
Real l2reg = 1e-6, maxent_l2reg = 1e-6;
int bptt = 5, bptt_period = 6;
Real rmsprop = -1;
Real gradient_clipping = 1;
bool learn_embeddings = true, learn_recurrent = true;
// - early stopping
Real bad_ratio = 1.003, awful_ratio = 0.997;
Real lr_decay_factor = 2;
int max_bad_epochs = 2;
// - nce
int nce_samples = 0;
std::string nce_maxent_model_weight_file;
Real nce_lnz = 9;
double nce_unigram_power = 1;
double nce_unigram_min_cells = 5;
};  // unnamed namespace

struct SimpleTimer;


struct TrainThreadTask {
  // local
  int chunk_id;
  int total_chunks;
  bool report, report_train_entopy;
  const SimpleTimer* timer;
  int epoch, n_inner_epochs;
  Real lrate, maxent_lrate;
  const std::string* train_file;

  // global
  uint64_t* seed;
  NNet* nnet;
  INoiseGenerator* noise_generator;
  int64_t* n_done_words;
  int64_t* n_done_bytes;
};


struct SimpleTimer {
  SimpleTimer() { Reset(); }

#ifdef NORT
  void Reset() { gettimeofday(&start, NULL); }

  double Tick() const {
    timeval finish;
    gettimeofday(&finish, NULL);
    double elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_usec - start.tv_usec) / 1000000000.0;
    return elapsed;
  }

  timeval start;
#else
  void Reset() { clock_gettime(CLOCK_MONOTONIC, &start); }

  double Tick() const {
    struct timespec finish;
    clock_gettime(CLOCK_MONOTONIC, &finish);
    double elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    return elapsed;
  }

  struct timespec start;
#endif
};

// Fill ngram_hashes with indices of maxent hashes for given position in the sentence
// Returns number of indices
inline int CalculateMaxentHashIndices(
    const NNet* nnet, const WordIndex* sen, int pos, uint64_t* ngram_hashes) {
  int maxent_present = CalculateMaxentHashIndices(
    sen, pos, nnet->cfg.maxent_order, nnet->cfg.maxent_hash_size - nnet->vocab.size(),
    kMaxentAddPadding, ngram_hashes);
  return maxent_present;
}

inline void PropagateForward(NNet* nnet, const WordIndex* sen, int sen_length, IRecUpdater* layer) {
  RowMatrix& input = layer->GetInputMatrix();
  for (int i = 0; i < sen_length; ++i) {
    input.row(i) = nnet->embeddings.row(sen[i]);
  }
  layer->ForwardSequence(sen_length);
}

// Checks file existence
bool Exists(const std::string& fname) {
  FILE *f = fopen(fname.c_str(), "rb");
  if (f == NULL) {
    return false;
  }
  fclose(f);
  return true;
}

// Returns entropy of the model in bits
//
// if print_logprobs is true, -log10(Prob(sentence)) is printed for each sentence in the file
Real EvaluateLM(NNet* nnet, const std::string& filename, bool print_logprobs, bool accurate_nce) {
  IRecUpdater* rec_layer_updater = nnet->rec_layer->CreateUpdater();
  bool kAutoInsertUnk = (kOOVPolicy == kConvertToUnk);
  SentenceReader reader(nnet->vocab, filename, nnet->cfg.reverse_sentence, kAutoInsertUnk);
  Real logprob_sum = 0;
  uint64_t n_words = 0;

  std::vector<Real> logprob_per_pos;
  if (nnet->cfg.use_nce && nnet->use_cuda) {
    nnet->nce->UploadNetWeightsToCuda(&nnet->maxent_layer);
  }
  while (reader.Read()) {
    if (reader.HasOOVWords() && kOOVPolicy == kSkipSentence) {
        if (print_logprobs)
            printf("OOV\n");
        continue;
    }

    const WordIndex* sen = reader.sentence();
    int seq_length = reader.sentence_length();
    Real sen_logprob = 0.0;

    PropagateForward(nnet, sen, seq_length, rec_layer_updater);

    const RowMatrix& output = rec_layer_updater->GetOutputMatrix();
    if (!nnet->cfg.use_nce) {
      // Hierarchical Softmax
      for (int target = 1; target <= seq_length; ++target) {
        uint64_t ngram_hashes[MAX_NGRAM_ORDER];
        int maxent_present = CalculateMaxentHashIndices(nnet, sen, target, ngram_hashes);
        const Real logprob = nnet->softmax_layer->CalculateLog10Probability(
            sen[target], ngram_hashes, maxent_present, kHSMaxentPrunning,
            output.row(target - 1).data(), &nnet->maxent_layer);

        sen_logprob -= logprob;
      }
    } else {
      // Noise Contrastive Estimation
      // We use batch logprob calculation to improve performance on GPU
      std::vector<uint64_t> ngram_hashes_all(seq_length * MAX_NGRAM_ORDER);
      std::vector<int> ngram_present_all(seq_length);

      for (int target = 1; target <= seq_length; ++target) {
        uint64_t* ngram_hashes = ngram_hashes_all.data() + MAX_NGRAM_ORDER * (target - 1);
        int maxent_present = CalculateMaxentHashIndices(nnet, sen, target, ngram_hashes);
        ngram_present_all[target - 1] = maxent_present;
      }

      nnet->nce->CalculateLog10ProbabilityBatch(
          output, &nnet->maxent_layer,
          ngram_hashes_all.data(), ngram_present_all.data(),
          sen, seq_length, !accurate_nce,
          &logprob_per_pos);
      for (int i = 0; i < seq_length; ++i) {
        sen_logprob -= logprob_per_pos[i];
      }
    }
    n_words += seq_length;
    logprob_sum += sen_logprob;
    if (print_logprobs) {
      printf("%f\n", -sen_logprob);
    }
  }

  delete rec_layer_updater;
  Real entropy = logprob_sum / log10(2) / n_words;
  return entropy;
}

void *RunThread(void *ptr) {
  TrainThreadTask& task = *reinterpret_cast<TrainThreadTask*>(ptr);
  NNet* nnet = task.nnet;
  IRecUpdater* rec_layer_updater = nnet->rec_layer->CreateUpdater();
  NCE::Updater* nce_updater = nnet->cfg.use_nce ? (new NCE::Updater(nnet->nce)) : NULL;

  Real train_logprob = 0;
  bool kAutoInsertUnk = false;
  SentenceReader reader(nnet->vocab, *task.train_file, nnet->cfg.reverse_sentence, kAutoInsertUnk);
  reader.SetChunk(task.chunk_id, task.total_chunks);
  int64_t n_done_bytes_local = 0, n_total_bytes = reader.GetFileSize();
  int64_t n_done_words_local = 0, n_last_report_at = 0;

  uint64_t next_random = *task.seed;
  while (reader.Read()) {
    n_done_words_local += reader.sentence_length();
    if (n_done_words_local - n_last_report_at > kReportEveryWords) {
      {
        int64_t diff = n_done_words_local - n_last_report_at;
        *task.n_done_words += diff;
        n_last_report_at = n_done_words_local;
      }

      {
        int64_t n_new_bytes = reader.GetDoneByteCount() - n_done_bytes_local;
        n_done_bytes_local += n_new_bytes;
        *task.n_done_bytes += n_new_bytes;
      }

      if (task.report) {
        float done_percent = static_cast<float>(*task.n_done_bytes) /
            (n_total_bytes / task.n_inner_epochs + 1) * 100;
        float speed = *task.n_done_words / (task.timer->Tick() * 1000 + 1e-5);
        fprintf(stderr, "\rEpoch %2d  lr: %.2e/%.2e  progress: %6.2f%%  %.2f Kwords/sec  ",
            task.epoch, task.lrate, task.maxent_lrate, done_percent, speed);
        if (task.report_train_entopy) {
          float train_entropy = train_logprob / log10(2) / n_done_words_local;
          fprintf(stderr, "entropy (bits) train: %8.5f ", train_entropy);
        }
        fflush(stdout);
      }
    }

    if (reader.HasOOVWords())
      continue;

    // A sentence contains <s>, followed by (seq_length - 1) actual words, followed by </s>
    // Both <s> and </s> are mapped to zero
    const WordIndex* sen = reader.sentence();
    const int seq_length = reader.sentence_length();

    // Compute hidden layer for all words
    PropagateForward(nnet, sen, seq_length, rec_layer_updater);

    // Calculate criterion given hidden layers
    const RowMatrix& output = rec_layer_updater->GetOutputMatrix();
    RowMatrix& output_grad = rec_layer_updater->GetOutputGradMatrix();
    output_grad.topRows(seq_length).setZero();
    for (int target = 1; target <= seq_length; ++target) {
      const Real* output_row = output.row(target - 1).data();
      Real* output_grad_row = output_grad.row(target - 1).data();
      WordIndex target_word = sen[target];

      uint64_t ngram_hashes[MAX_NGRAM_ORDER] = {0};
      int maxent_present = CalculateMaxentHashIndices(nnet, sen, target, ngram_hashes);

      if (!nnet->cfg.use_nce) {
        const Real word_logprob = nnet->softmax_layer->PropagateForwardAndBackward(
            task.report_train_entopy, target_word, ngram_hashes, maxent_present,
            task.lrate, task.maxent_lrate, l2reg, maxent_l2reg, gradient_clipping,
            output_row, output_grad_row, &nnet->maxent_layer);
        if (task.report_train_entopy) {
          train_logprob -= word_logprob;
        }
      } else {
        NoiseSample sample;

        next_random = task.noise_generator->PrepareNoiseSample(
            next_random, nce_samples, sen, target, &sample);

        nce_updater->PropagateForwardAndBackward(
            output.row(target - 1), target_word, ngram_hashes, maxent_present,
            sample, task.lrate, l2reg, task.maxent_lrate, maxent_l2reg, gradient_clipping,
            output_grad.row(target - 1), &nnet->maxent_layer);
      }
    }

    rec_layer_updater->BackwardSequence(seq_length, GetNextRandom(&next_random), bptt_period, bptt);

    // Update embeddings
    if (learn_embeddings) {
      RowMatrix& input_grad = rec_layer_updater->GetInputGradMatrix();
      ClipMatrix(input_grad.topRows(seq_length), gradient_clipping);
      for (int input = seq_length - 1; input >= 0; --input) {
        WordIndex last_word = sen[input];
        nnet->embeddings.row(last_word) *= (1 - l2reg);
        nnet->embeddings.row(last_word).noalias() += input_grad.row(input) * task.lrate;
      }
    }

    // Update recurrent weights
    if (learn_recurrent) {
      rec_layer_updater->UpdateWeights(seq_length, task.lrate, l2reg, rmsprop, gradient_clipping);
    }
  }

  delete rec_layer_updater;
  delete nce_updater;
  *task.seed = next_random;
#ifndef NOTHREAD
  pthread_exit(NULL);
#else
  return NULL;
#endif
}

void TrainLM(
    const std::string& model_weight_file,
    const std::string& train_file, const std::string& valid_file,
    bool show_progress, bool show_train_entropy, int n_threads, int n_inner_epochs,
    NNet* nnet) {
  NNet* noise_net = NULL;
  INoiseGenerator* noise_generator = NULL;
  if (nnet->cfg.use_nce) {
    if (nce_maxent_model_weight_file[0] != 0) {
      const bool kUseCuda = true;
      const bool kUseCudaMemoryEfficient = true;
      noise_net = new NNet(nnet->vocab, nce_maxent_model_weight_file, kUseCuda, kUseCudaMemoryEfficient);
      if (noise_net->cfg.layer_size != 0) {
        fprintf(stderr, "ERROR: Cannot initialize HSMaxEntNoiseGenerator (layer size != 0)\n");
        exit(1);
      }

      {
        const bool kPrintLogprobs = false;
        const bool kNCEAccurate = true;
        Real test_enropy = EvaluateLM(noise_net, valid_file, kPrintLogprobs, kNCEAccurate);
        fprintf(stderr, "Noise Model Valid entropy %f\n", test_enropy);
      }

      noise_generator = new HSMaxEntNoiseGenerator(
        noise_net->softmax_layer, &noise_net->maxent_layer,
        noise_net->cfg.maxent_hash_size, nnet->vocab.size(),
        noise_net->cfg.maxent_order);
    } else {
      noise_generator = new UnigramNoiseGenerator(
        nnet->vocab, nce_unigram_power, nce_unigram_min_cells);
    }
  }

  std::vector<uint64_t> thread_seeds(n_threads);
  for (size_t i = 0; i < thread_seeds.size(); ++i) {
    thread_seeds[i] = i;
  }

#ifndef NOTHREAD
  std::vector<pthread_t> threads(n_threads);
#endif

  Real bl_entropy = -1;
  {
    const bool kPrintLogprobs = false;
    const bool kNCEAccurate = true;
    bl_entropy = EvaluateLM(nnet, valid_file, kPrintLogprobs, kNCEAccurate);
    fprintf(stderr, "Initial entropy (bits) valid: %8.5f\n", bl_entropy);
  }

  bool do_lr_decay = false;
  Real lrate = initial_lrate;
  Real maxent_lrate = initial_maxent_lrate;
  for (int n_bad_epochs = 0, epoch = 1; n_bad_epochs <= max_bad_epochs; ++epoch) {
    int inner_epoch = (epoch - 1) % n_inner_epochs;

    if (do_lr_decay) {
      lrate /= lr_decay_factor;
      maxent_lrate /= lr_decay_factor;
    }

    SimpleTimer timer;
    std::vector<TrainThreadTask> tasks(n_threads);
    int64_t n_done_words = 0;
    int64_t n_done_bytes = 0;
    for (int i = 0; i < n_threads; ++i) {
      tasks[i].chunk_id = i + inner_epoch * n_threads;
      tasks[i].total_chunks = n_inner_epochs * n_threads;
      tasks[i].report = (i == 0);
      tasks[i].report_train_entopy = show_train_entropy;
      tasks[i].timer = &timer;
      tasks[i].epoch = epoch;
      tasks[i].n_inner_epochs = n_inner_epochs;
      tasks[i].lrate = lrate;
      tasks[i].maxent_lrate = maxent_lrate;
      tasks[i].train_file = &train_file;

      tasks[i].seed = &thread_seeds[i];
      tasks[i].nnet = nnet;
      tasks[i].noise_generator = noise_generator;
      tasks[i].n_done_words = &n_done_words;
      tasks[i].n_done_bytes = &n_done_bytes;
    }
    for (int i = 0; i < n_threads; i++) {
#ifdef NOTHREAD
      RunThread(reinterpret_cast<void*>(&tasks[i]));
#else
      pthread_create(&threads[i], NULL, RunThread, reinterpret_cast<void*>(&tasks[i]));
    }
    for (int i = 0; i < n_threads; i++) {
      pthread_join(threads[i], NULL);
#endif
    }
    const double elapsed_train = timer.Tick();

    const bool kPrintLogprobs = false;
    const bool kNCEAccurate = true;
    const Real entropy = EvaluateLM(nnet, valid_file, kPrintLogprobs, kNCEAccurate);
    if (!show_progress) {
      fprintf(stderr, "Epoch %d ", epoch);
    }
    if (!show_train_entropy) {
      fprintf(stderr, "entropy (bits) ");
    }
    fprintf(stderr, "valid: %8.5f", entropy);

    const double elapsed = timer.Tick();
    const double elapsed_validate = elapsed - elapsed_train;
    if (elapsed_train < 999) {
      fprintf(stderr, "  elapsed: %.1lfs+%.1lfs", elapsed_train, elapsed_validate);
    } else if (elapsed < 999 * 60) {
      fprintf(stderr, "  elapsed: %.1lfm+%.1lfm", elapsed_train / 60, elapsed_validate / 60);
    } else {
      fprintf(stderr, "  elapsed: %.1lfh+%.1lfh", elapsed_train / 3600, elapsed_validate / 3600);
    }

    Real ratio = bl_entropy / entropy;
    if (isnan(entropy) || isinf(entropy) || !(ratio >= bad_ratio)) {
      // !(ratio >= bad_ratio) will catch nan and inf even in fastmath mode
      if (isnan(entropy) || isinf(entropy) || !(ratio >= awful_ratio)) {
        fprintf(stderr, "\tAwful: Nnet rejected");
        nnet->ReLoad(model_weight_file);
      } else {
        fprintf(stderr, "\tBad: ");
        if (do_lr_decay || max_bad_epochs == 0) {
          fprintf(stderr, "%d more to stop", max_bad_epochs - n_bad_epochs);
        } else {
          fprintf(stderr, "start lr decay");
        }
      }
      ++n_bad_epochs;
      do_lr_decay = true;
    }
    fprintf(stderr, "\n");

    if (ratio >= awful_ratio) {
      nnet->Save(model_weight_file);
      bl_entropy = entropy;
    }
  }

  if (nnet->cfg.use_nce) {
    delete noise_generator;
    delete noise_net;
  }
}


void SampleFromLM(NNet* nnet, int seed, int n_samples, Real generate_temperature) {
  std::vector<WordIndex> wids;
  {
    char buffer[MAX_STRING];
    for (WordReader reader(""); reader.ReadWord(buffer);) {
      WordIndex wid = nnet->vocab.GetIndexByWord(buffer);
      if (wid == 0) {
        break;
      }
      if (wid == Vocabulary::kWordOOV) {
        wid = nnet->vocab.GetIndexByWord("<unk>");
        if (wid == Vocabulary::kWordOOV) {
          fprintf(stderr, "ERROR Word '%s' is not found in vocabulary;"
                 " moreover, <unk> is not found as well\n", buffer);
          exit(1);
        }
      }
      wids.push_back(wid);
    }
  }
  printf("Generating with seed:");
  for (size_t i = 0; i < wids.size(); ++i) {
    printf(" %s", nnet->vocab.GetWordByIndex(wids[i]));
  }
  printf("\n");

  printf("Format: <prefix> | <generated> | <log10prob(generated)> | <log10prob per word>\n");

  srand(seed);

  std::vector<double> probs(nnet->vocab.size());
  IRecUpdater* updater = nnet->rec_layer->CreateUpdater();
  PropagateForward(nnet, wids.data(), wids.size(), updater);
  for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
    for (size_t i = 0; i < wids.size(); ++i) {
      printf("%s ", nnet->vocab.GetWordByIndex(wids[i]));
    }
    printf("| ");

    std::vector<WordIndex> sen = wids;

    double ending_log10prob = 0;
    for (; sen.back() != 0;) {
      sen.push_back(0);
      int target = sen.size() - 1;
      const RowMatrix& output = updater->GetOutputMatrix();
      RowMatrix& input = updater->GetInputMatrix();

      // Calculate (unnormalized) probabilities for each word to follow
      if (!nnet->cfg.use_nce) {
        for (int wid = 0; wid < nnet->vocab.size(); ++wid) {
          sen.back() = wid;
          uint64_t ngram_hashes[MAX_NGRAM_ORDER];
          bool kDynamicMaxentPruning = false;
          int maxent_present = CalculateMaxentHashIndices(nnet, sen.data(), target, ngram_hashes);
          probs[wid] = pow(10., nnet->softmax_layer->CalculateLog10Probability(
              sen[target], ngram_hashes, maxent_present, kDynamicMaxentPruning,
              output.row(target - 1).data(), &nnet->maxent_layer) / generate_temperature);

        }
      } else {
        for (int wid = 0; wid < nnet->vocab.size(); ++wid) {
          sen.back() = wid;
          uint64_t ngram_hashes[MAX_NGRAM_ORDER];
          int maxent_present = CalculateMaxentHashIndices(nnet, sen.data(), target, ngram_hashes);
          probs[wid] = exp(nnet->nce->CalculateWordLnScore(
              output.row(target - 1), &nnet->maxent_layer,
              ngram_hashes, maxent_present,
              sen[target]) / generate_temperature);
        }
      }

      {
        // Calcluate normalization constant
        double s = 0;
        for (int wid = nnet->vocab.size(); wid-- > 0; ) {
          // Sum in reverse order to improve accuracy
          s += probs[wid];
        }

        // Sample threshold
        double p = static_cast<double>(rand()) / RAND_MAX;
        p *= s;

        std::vector<int> order(nnet->vocab.size());
        for (int wid = 0; wid < nnet->vocab.size(); ++wid) {
            order[wid] = wid;
        }
        std::random_shuffle(order.begin(), order.end());

        // Find the word that corresponds to the threshold
        int wid;
        for (wid = 0; p >= 0 && wid < nnet->vocab.size(); ++wid) {
          p -= probs[order[wid]];
        }
        wid -= 1;
        sen.back() = order[wid];

        ending_log10prob += log10(probs[order[wid]]) - log10(s);
      }

      printf("%s ", nnet->vocab.GetWordByIndex(sen.back()));
      fflush(stdout);
      input.row(sen.size() - 1) = nnet->embeddings.row(sen.back());
      updater->ForwardStep(sen.size() - 1);
    }

    printf("| %f", ending_log10prob);
    printf(" | %f", ending_log10prob / std::max<int>(1, sen.size() - wids.size()));

    printf("\n");
  }
  delete updater;
}

int main(int argc, char **argv) {
#ifdef DETECT_FPE
  feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT & ~FE_UNDERFLOW);
#endif

  std::string layer_type = "sigmoid";
  int layer_size = 100, maxent_order = 0, random_seed = 0, hs_arity = 2;
  uint64_t maxent_hash_size = 0;
  int layer_count = 1;
  std::string model_vocab_file, test_file, train_file, valid_file;
  bool use_cuda = kHaveCudaSupport;
  bool use_cuda_memory_efficient = false;
  bool reverse_sentence = false;
  bool show_progress = true, show_train_entropy = false;
  int n_threads = 1;
  int n_inner_epochs = 1;
  bool nce_accurate_test = false;
  Real diagonal_initialization = 0;
  int n_samples = 0;
  Real generate_temperature = 1;
  int bptt_skip = bptt_period - bptt;

  SimpleOptionParser opts;
  opts.Echo("Fast Recurrent Neural Network Language Model");
  opts.Echo("Main options:");
  opts.Add("rnnlm", "Path to model file (mandatory)", &model_vocab_file);
  opts.Add("train", "Train file", &train_file);
  opts.Add("valid", "Validation file (used for early stopping)", &valid_file);
  opts.Add("generate-samples", "Number of sentences to generate in sampling mode", &n_samples);
  opts.Add("generate-temperature", "Softmax temperature (use lower values to get robuster results)", &generate_temperature);
  opts.Add("hidden", "Size of embedding and hidden layers", &layer_size);
  opts.Add("hidden-count", "Count of hidden layers; all hidden layers have the same type and size", &layer_count);
  opts.Add("hidden-type", "Hidden layer activation (sigmoid, tanh, relu, gru, gru-bias, gru-insyn, gru-full)", &layer_type);
  opts.Add("arity", "Arity of the HS tree", &hs_arity);
  opts.Add("direct", "Size of maxent layer in millions", &maxent_hash_size);
  opts.Add("direct-order", "Maximum order of ngram features", &maxent_order);
  opts.Add("test", "Test file; if not empty, evaluation mode is enabled, i.e. no training", &test_file);
  opts.Add("epoch-per-file", "Treat one pass over the train file as given number of epochs (usefull for big datasets)", &n_inner_epochs);
  opts.Add("seed", "Random seed for weight initialization and sampling", &random_seed);
  opts.Add("threads", "Number of threads to use; optimal value is the number of physical cores on a CPU", &n_threads);
  opts.Add("reverse-sentence", "Predict sentence words in reversed order", &reverse_sentence);
  opts.Add("show-progress", "Show training progress", &show_progress);
  opts.Add("show-train-entropy", "Show average entropy on train set for the first thread; doesn't work for NCE", &show_train_entropy);
  opts.Add("diagonal-initialization", "Initialize recurrent matrix with x * I (x is the value and I is identity matrix); must be greater then zero to have any effect", &diagonal_initialization);
  opts.Echo();
  opts.Echo("Optimization options:");
  opts.Add("rmsprop", "RMSprop coefficient; rmsprop<0 disables RMSProp and rmsprop=0 equivalent to RMS", &rmsprop);
  opts.Add("gradient-clipping", "Clip updates above the value", &gradient_clipping);
  opts.Add("learn-recurrent", "Learn hidden layer weights", &learn_recurrent);
  opts.Add("learn-embeddings", "Learn embedding weights", &learn_embeddings);
  opts.Add("bptt", "Length of truncated BPTT unfolding; set to zero to back-propagate through entire sentence", &bptt);
  opts.Add("bptt-skip", "Number of steps without BPTT; doesn't have any effect if bptt is 0", &bptt_skip);
  opts.Add("alpha", "Learning rate for recurrent and embedding weights", &initial_lrate);
  opts.Add("maxent-alpha", "Learning rate for maxent layer", &initial_maxent_lrate);
  opts.Add("beta", "Weight decay for recurrent and embedding weight, i.e. L2-regularization", &l2reg);
  opts.Add("maxent-beta", "Weight decay for maxent layer, i.e. L2-regularization", &maxent_l2reg);
  opts.Echo();
  opts.Echo("Early stopping options (let `ratio' be a ratio of previous epoch validation entropy to new one):");
  opts.Add("stop", "If `ratio' less than `stop' then start leaning rate decay", &bad_ratio);
  opts.Add("lr-decay-factor", "Learning rate decay factor", &lr_decay_factor);
  opts.Add("reject-threshold", "If (whats more) `ratio' less than `reject-threshold' then purge the epoch", &awful_ratio);
  opts.Add("retry", "Stop training once `ratio' has hit `stop' at least `retry' times", &max_bad_epochs);
  opts.Echo();
  opts.Echo("Noise Contrastive Estimation options:");
  opts.Add("nce", "Number of noise samples; if nce is position then NCE is used instead of HS", &nce_samples);
  opts.Add("nce-accurate-test", "Explicitly normalize output probabilities; use this option to compute actual entropy", &nce_accurate_test);
  opts.Add("use-cuda", "Use CUDA to compute validation entropy and test entropy in accurate mode, i.e. if nce-accurate-test is true", &use_cuda);
  opts.Add("use-cuda-memory-efficient", "Do not copy the whole maxent layer on GPU. Slower, but could be useful to deal with huge maxent layers", &use_cuda_memory_efficient);
  opts.Add("nce-unigram-power", "Discount power for unigram frequency", &nce_unigram_power);
  opts.Add("nce-lnz", "Ln of normalization constant", &nce_lnz);
  opts.Add("nce-unigram-min-cells", "Minimum number of cells for each word in unigram table (works akin to Laplacian smoothing)", &nce_unigram_min_cells);
  opts.Add("nce-maxent-model", "Use given the model as a noise generator; the model must a pure maxent model trained by the program", &nce_maxent_model_weight_file);
  opts.Echo();
  opts.Echo();
  opts.Echo("How to");
  opts.Echo("  train model with Hierarchical Softmax:");
  opts.Echo("    ./rnnlm -rnnlm model_name -train train.txt -valid validation.txt -hidden 100");
  opts.Echo("  train model with Noise Contrastive Estimation:");
  opts.Echo("    ./rnnlm -rnnlm model_name -train train.txt -valid validation.txt -hidden 100 -nce 22");
  opts.Echo("  apply model:");
  opts.Echo("    ./rnnlm -rnnlm model_name -test test.txt -nce-accurate-test 1");
  opts.Echo();
  opts.Echo("All text corpora must be splitted by sentences (one sentence per line) and tokenized (separated by whitespaces");
  // Ignore these flags for back compability
  opts.Ignore("-nbest");
  opts.Ignore("-independent");
  // Add alias for back compability
  opts.AddAlias("bptt-block", "bptt-skip");

  if (argc == 1) {
    opts.PrintHelp();
    return 0;
  }
  opts.Parse(argc, argv);

  bptt_period = bptt + bptt_skip;
  if (model_vocab_file.empty()) {
    fprintf(stderr, "ERROR model file argument (-rnnlm) is required\n");
    return 1;
  }
  if (test_file.empty() && train_file.empty() && n_samples == 0) {
    fprintf(stderr, "ERROR you must provide either train file or test file\n");
    return 1;
  }

  if (maxent_hash_size == 0 || maxent_order == 0) {
    maxent_hash_size = 0;
    maxent_order = 0;
  } else if (maxent_order > MAX_NGRAM_ORDER) {
    fprintf(stderr, "ERROR maxent_order must be less than or equal to %d\n", MAX_NGRAM_ORDER);
    return 1;
  }
  if (bad_ratio < awful_ratio) {
    fprintf(stderr, "ERROR Value for -stop ratio must be less or equal to -reject-threshold ratio\n");
    return 1;
  }

  if (!nce_accurate_test && !test_file.empty()) {
    use_cuda = false;
  }

  srand(random_seed);

  // Construct/load vocabulary
  Vocabulary vocab;
  const bool has_vocab = Exists(model_vocab_file);
  if (has_vocab) {
    vocab.Load(model_vocab_file);
    fprintf(stderr, "Read the vocabulary: %d words\n", vocab.size());
  } else {
    vocab.BuildFromCorpus(train_file, show_progress);
    vocab.AdjustSizeForSoftmaxTree(hs_arity);
    vocab.Dump(model_vocab_file);
    fprintf(stderr, "Constructed a vocabulary: %d words\n", vocab.size());
  }

  // Construct/load neural network
  const std::string model_weight_file = model_vocab_file + ".nnet";
  NNet* main_nnet = NULL;
  if (has_vocab && Exists(model_weight_file)) {
    fprintf(stderr, "Restoring existing nnet\n");
    main_nnet = new NNet(vocab, model_weight_file, use_cuda, use_cuda_memory_efficient);
  } else {
    fprintf(stderr, "Constructing a new net (no model file is found)\n");
    if (maxent_hash_size) {
      // maxent option stores the size of the hash in millions
      maxent_hash_size *= 1000 * 1000;
      // for back-compability
      maxent_hash_size -= maxent_hash_size % vocab.size();
    }
    NNetConfig cfg = {
      layer_size, layer_count, maxent_hash_size, maxent_order,
      (nce_samples > 0), static_cast<Real>(nce_lnz), reverse_sentence,
      hs_arity, layer_type};
    main_nnet = new NNet(vocab, cfg, use_cuda, use_cuda_memory_efficient);
    if (diagonal_initialization > 0) {
      main_nnet->ApplyDiagonalInitialization(diagonal_initialization);
    }
    main_nnet->Save(model_weight_file);
  }

  if (show_train_entropy && (main_nnet->cfg.use_nce > 0 || main_nnet->cfg.maxent_order > 0)) {
    fprintf(stderr, "WARNING --show-train-entropy could be used only for HS based models without maxent\n");
    show_train_entropy = false;
  }

  if (n_samples > 0) {
    SampleFromLM(main_nnet, random_seed, n_samples, generate_temperature);
  } else if (!test_file.empty()) {
    // Apply mode
    const bool kPrintLogprobs = true;
    Real test_enropy = EvaluateLM(main_nnet, test_file, kPrintLogprobs, nce_accurate_test);
    if (!main_nnet->cfg.use_nce || nce_accurate_test) {
      fprintf(stderr, "Test entropy %f\n", test_enropy);
    } else {
      fprintf(stderr, "Use -nce-accurate-test to calculate entropy\n");
    }
  } else {
    // Train mode
    TrainLM(
        model_weight_file, train_file, valid_file,
        show_progress, show_train_entropy, n_threads, n_inner_epochs, main_nnet);
  }

  delete main_nnet;
  return 0;
}
