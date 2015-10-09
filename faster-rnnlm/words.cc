#include "faster-rnnlm/words.h"

#include <stdlib.h>
#include <string.h>

#include <algorithm>

namespace {
const char kEOSTag[] = "</s>";
const int kEOSSize = sizeof(kEOSTag);
const size_t kHashMinSize = 100000;
const double kHashMinFactor = 0.5;
const double kHashMaxFactor = 0.8;
};  // unnamed namespace

const WordIndex Vocabulary::kWordOOV;


// ============================================================================
// ============================== WordReader ==================================
// ============================================================================

inline bool IsSpace(char c) {
  return c == ' ' || c == '\r' || c == '\t' ||  c == '\n';
}

void FseekOrDie(FILE* stream, long int offset, int whence, const std::string& fname) {
  if (fseek(stream, offset, whence) != 0) {
    fprintf(
        stderr, "ERROR: Failed to seek over file '%s'."
       " Make sure it is a regular file", fname.c_str());
    exit(1);
  }
}

// Read space-serated words from file; adds </s> to the end of each line
WordReader::WordReader(const std::string& fname)
    : file_(fname.empty() ? stdin : fopen(fname.c_str(), "rb"))
    , pointer_(NULL)
    , buffer_(new char[MAX_LINE_SIZE])
    , fname_(fname)
    , file_size_(-1)
    , chunk_start_(0)
    , chunk_end_(-1)
{
  if (file_ == NULL) {
    fprintf(stderr, "ERROR failed to open %s\n", fname.c_str());
  }
  if (!fname.empty()) {
    FseekOrDie(file_, 0, SEEK_END, fname_);
    file_size_ = ftell(file_);
    FseekOrDie(file_, 0, SEEK_SET, fname_);
  }
}

bool WordReader::ReadWord(char* word) {
  if (pointer_ == NULL) {
    // loop until EOF or not-empty line
    for (; pointer_ == NULL || *pointer_ == 0;) {
      if (chunk_end_ != -1 && chunk_end_ < ftell(file_)) {
        return false;
      }

      if (fgets(buffer_, MAX_LINE_SIZE, file_) == NULL) {
        return false;
      }

      for (pointer_ = buffer_; IsSpace(*pointer_); ++pointer_) {}
    }
  }
  for (; IsSpace(*pointer_); ++pointer_) {}

  if (*pointer_ == 0) {
    strncpy(word, kEOSTag, MAX_STRING - 1);
    word[std::min(MAX_STRING, kEOSSize - 1)] = 0;
    pointer_ = NULL;
  } else {
    int i = 0;
    for (; !IsSpace(*pointer_) && i < MAX_STRING - 1; ++pointer_, ++i) {
      word[i] = *pointer_;
    }
    word[i] = 0;

    // drop reduntant characters
    for (; !IsSpace(*pointer_); ++pointer_) {}
  }

  return true;
}


void WordReader::SetChunk(int chunk, int chunk_count) {
  chunk_start_ = file_size_ / chunk_count * chunk;
  FseekOrDie(file_, chunk_start_, SEEK_SET, fname_);
  if (chunk + 1 == chunk_count) {
    chunk_end_ = -1;
  } else {
    chunk_end_ = file_size_ / chunk_count * (chunk + 1);
  }
  if (chunk != 0) {
    // skipping to the next newline
    for (char tmp[MAX_STRING]; ReadWord(tmp) && strcmp(tmp, kEOSTag) != 0; ) {}
  }
}

int64_t WordReader::GetDoneByteCount() const {
  return ftell(file_) - chunk_start_;
}

// ============================================================================
// ============================== Vocabulary ==================================
// ============================================================================


class Vocabulary::HashImpl {
 public:
  explicit HashImpl(const Vocabulary& vocabulary) : vocabulary(vocabulary) {
    Rebuild();
  }

  WordIndex Get(const char* word) const {
    return hash2index[Find(word)];
  }

  void Insert(const char* word, WordIndex idx) {
    size_t hash_idx = Find(word);
    if (hash2index[hash_idx] == Vocabulary::kWordOOV) {
      hash2index[hash_idx] = idx;
      if (hash2index.size() * kHashMaxFactor < vocabulary.size()) {
        Rebuild();
      }
    }
  }

  void Rebuild() {
    size_t size = std::max(kHashMinSize, static_cast<size_t>(vocabulary.size() / kHashMinFactor));
    hash2index.resize(size);
    std::fill(hash2index.begin(), hash2index.end(), Vocabulary::kWordOOV);
    for (int i = 0; i < vocabulary.size(); ++i) {
      Insert(vocabulary.GetWordByIndex(i), i);
    }
  }

 private:
  std::vector<WordIndex> hash2index;
  const Vocabulary& vocabulary;

  size_t Find(const char* word) const {
    size_t hash_index = CalculateHash(word) % hash2index.size();
    for (;; ++hash_index) {
      if (hash_index >= hash2index.size()) {
        hash_index = 0;
      }
      if (hash2index[hash_index] == Vocabulary::kWordOOV) {
        return hash_index;
      }
      WordIndex wid = hash2index[hash_index];
      if (strcmp(word, vocabulary.GetWordByIndex(wid)) == 0) {
        return hash_index;
      }
    }
  }

  static uint64_t CalculateHash(const char* word) {
    uint64_t hash = 42;
    const uint64_t p = 1000 * 1000 * 1000 + 7;
    for (; *word != 0; ++word) {
      hash = hash * p + *word;
    }
    return hash;
  }
};

Vocabulary::Vocabulary() : hash_impl_(new HashImpl(*this))
{
  // empty constructor
}

Vocabulary::~Vocabulary() {
  delete hash_impl_;
  for (int a = 0; a < size(); a++) {
    delete[] words_[a].word;
  }
}

WordIndex Vocabulary::GetIndexByWord(const char *word) const {
  return hash_impl_->Get(word);
}

const char* Vocabulary::GetWordByIndex(WordIndex index) const {
  if (static_cast<int>(index) >= 0 && static_cast<int>(index) < size()) {
    return words_[index].word;
  }
  return NULL;
}

WordIndex Vocabulary::AddWord(const char *word) {
  int length = std::min<int>(strlen(word) + 1, MAX_STRING);

  Vocabulary::Word vocab_word = {0, NULL};
  vocab_word.word = new char[length];
  strncpy(vocab_word.word, word, length - 1);
  vocab_word.word[length - 1] = 0;
  words_.push_back(vocab_word);

  hash_impl_->Insert(word, size() - 1);
  return size() - 1;
}

struct VocabularyWordComparator {
  const bool stable_sort;

  explicit VocabularyWordComparator(bool stable_sort) : stable_sort(stable_sort) {}

  bool operator()(const Vocabulary::Word& first, const Vocabulary::Word& second) const {
    int frequency_diff = first.freq - second.freq;
    if (stable_sort || frequency_diff != 0) {
        return frequency_diff > 0;
    }
    return strncmp(first.word, second.word, MAX_STRING) < 0;
  }
};

// Sorts all words (except for '</s>') by frequency
//
// To sorting mode are available
//   stable mode (stable=true), i.e. stable sort by frequency (desc)
//   non-stable mode (stable=false), i.e. sort by frequency (desc) and word (asc)
//
// Note that word '</s>' has index zero
void Vocabulary::Sort(bool stable) {
  std::stable_sort(words_.begin() + 1, words_.end(), VocabularyWordComparator(stable));
  hash_impl_->Rebuild();
}

void Vocabulary::BuildFromCorpus(const std::string& fpath, bool show_progress) {
  char buffer[MAX_STRING];
  AddWord(kEOSTag);

  WordReader reader(fpath.c_str());
  for (int64_t read_words = 0; reader.ReadWord(buffer); ++read_words) {
    if (show_progress && (read_words % 1000000 == 0)) {
      fprintf(stderr, "Reading train file: %.1lfKK words\r",
          static_cast<double>(read_words / 1000) / 1000);
    }
    WordIndex wid = GetIndexByWord(buffer);
    if (wid == kWordOOV) {
      wid = AddWord(buffer);
    }
    words_[wid].freq++;
  }
  Sort(false);
}

void Vocabulary::Dump(const std::string& fpath) const {
  FILE *file = fopen(fpath.c_str(), "wb");
  if (file == NULL) {
    fprintf(stderr, "ERROR: Cannot open vocabulary file fot writing '%s'\n", fpath.c_str());
    exit(1);
  }
  for (int i = 0; i < size(); i++) {
    fprintf(file, "%s %" PRId64 "\n", words_[i].word, words_[i].freq);
  }
  fclose(file);
}

void Vocabulary::Load(const std::string& fpath) {
  FILE *file = fopen(fpath.c_str(), "rb");
  if (file == NULL) {
    fprintf(stderr, "ERROR: Cannot find vocabulary file '%s'\n", fpath.c_str());
    exit(1);
  }

  for (int line_number = 0; !feof(file); ++line_number) {
    char buffer[MAX_STRING];
    uint64_t count;
    if (fscanf(file, "%s %" PRId64 " ", buffer, &count) != 2) {
      fprintf(stderr, "WARNING: Skipping ill-formed line #%d in the vocabulary\n", line_number);
      continue;
    }
    WordIndex wid = AddWord(buffer);
    words_[wid].freq = count;
  }
  Sort(true);
}

void Vocabulary::AdjustSizeForSoftmaxTree(int arity) {
  int imperfection_size = (size() - 1 + (arity - 1)) % (arity - 1);
  if (imperfection_size == 0) {
      return;
  }
  fprintf(stderr, "Have to add %d words\n", imperfection_size);
  const int kBufferSize = 100;
  char buffer[kBufferSize];
  for (int a = 0; a < arity - 1 - imperfection_size; ++a) {
      snprintf(static_cast<char*>(buffer), kBufferSize,  "__**FAKE_WORD%d", a);
      WordIndex wid = AddWord(buffer);
      words_[wid].freq = 0;
  }
  Sort(false);
}


// ============================================================================
// =========================== SentenceReader =================================
// ============================================================================

SentenceReader::SentenceReader(
        const Vocabulary& vocab, const std::string& filename, bool reverse, bool auto_insert_unk)
    : WordReader(filename)
    , sentence_length_(0)
    , sentence_id_(-1)
    , vocab_(vocab)
    , unk_word_(vocab.GetIndexByWord("<unk>"))
    , reverse_(reverse)
    , done_(false)
    , oov_occured_(false)
{
  if (auto_insert_unk && unk_word_ == Vocabulary::kWordOOV) {
    fprintf(stderr, "Cannot use auto_insert_unk as '<unk>' is not found in the vocabulary\n");
    exit(1);
  }
}

bool SentenceReader::ReadWordId(WordIndex* wid) {
  char buffer[MAX_STRING];
  if (!ReadWord(buffer)) {
    return false;
  }
  *wid = vocab_.GetIndexByWord(buffer);
  return true;
}

bool SentenceReader::Read() {
  if (done_)
    return false;

  ++sentence_id_;
  sen_[0] = 0;      // <s> token
  oov_occured_ = false;
  for (sentence_length_ = 1; sentence_length_ < MAX_SENTENCE_WORDS; ++sentence_length_) {
    WordIndex wid;
    if (!ReadWordId(&wid)) {
      done_ = true;
      return false;
    }
    if (wid == Vocabulary::kWordOOV) {
      oov_occured_ = true;
      wid = unk_word_;
    }
    sen_[sentence_length_] = wid;
    if (wid == 0) {
      break;
    }
  }

  if (sentence_length_ == MAX_SENTENCE_WORDS) {
    // too long sentence -> drop trailing words
    for (;;) {
      WordIndex wid;
      if (!ReadWordId(&wid)) {
        done_ = true;
        return false;
      }
      if (wid == 0) {
        break;
      }
    }
    --sentence_length_;
    sen_[sentence_length_] = 0;
  }

  if (reverse_) {
    for (int i = 1; i < sentence_length_ - i; ++i) {
      std::swap(sen_[i], sen_[sentence_length_ - i]);
    }
  }

  return true;
}


