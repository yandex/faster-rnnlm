#ifndef FASTER_RNNLM_WORDS_H_
#define FASTER_RNNLM_WORDS_H_

#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>

#include <string>
#include <vector>

#include "faster-rnnlm/settings.h"


class Vocabulary {
 public:
  static const WordIndex kWordOOV = -1;

  Vocabulary();

  ~Vocabulary();

  int size() const { return words_.size(); }

  // Returns index of the word in the vocabulary; if the word is not found, returns kWordOOV
  WordIndex GetIndexByWord(const char* word) const;

  // Returns a word by its index; for invalid indices NULL pointer is returned
  const char* GetWordByIndex(WordIndex index) const;

  // Returns a frequence of a word by its index; for invalid indices - UB
  int64_t GetWordFrequency(WordIndex index) const { return words_[index].freq; }

  // Reads the vocabulary from the file
  void Load(const std::string& fpath);

  // Saves the vocabulary to the file
  void Dump(const std::string& fpath) const;

  // Reads corpus and stores all words with their frequencies
  void BuildFromCorpus(const std::string& fpath, bool show_progress);

  // Adds a few fake words if that is needed to build a full k-ary tree
  void AdjustSizeForSoftmaxTree(int arity);

  struct Word {
    int64_t freq;
    char *word;
  };

 private:
  class HashImpl;

  std::vector<Word> words_;
  HashImpl* hash_impl_;

  // Adds a word to the vocabulary with zero frequency; returns word index
  WordIndex AddWord(const char *word);

  void Sort(bool stable);
};


// Reads space-serated words from a file
//  - adds </s> to end of each line
//  - empty lines are skipped
class WordReader {
 public:
  explicit WordReader(const std::string& fname);

  ~WordReader() { if (file_ != stdin) fclose(file_); delete[] buffer_; }

  // Reads a word; returns false iff EOF
  // Words longer than MAX_STRING are truncated
  bool ReadWord(char* word);

  // Forces the reader to iterate over only a 1 / chunk_count part of the training file
  void SetChunk(int chunk, int chunk_count);

  int64_t GetFileSize() const { return file_size_; }

  // amount of bytes that were read from the start of the chunk
  int64_t GetDoneByteCount() const;

 protected:
  FILE* file_;
  char* pointer_;
  char* buffer_;
  const std::string fname_;
  int64_t file_size_, chunk_start_, chunk_end_;
};


// SentenceReader class represents a corpora as a sequence of sentences,
// where each sentence is an array of word indices
class SentenceReader : public WordReader {
 public:
  SentenceReader(const Vocabulary& vocab, const std::string& filename, bool reverse, bool auto_insert_unk);

  // Tries to read next sentence; returns false iff eof or end-of-chunk
  bool Read();

  // Returns pointer to the last read sentence
  const WordIndex* sentence() const { return sen_; }

  // Returns the length of the last read sentence
  int sentence_length() const { return sentence_length_; }

  // Returns number of sentences read so far
  int64_t sentence_id() const { return sentence_id_; }

  // Returns true iff the last line has any OOV words, that were mapped to unk_word
  bool HasOOVWords() const { return oov_occured_; }

 private:
  SentenceReader(SentenceReader&);
  SentenceReader& operator=(SentenceReader&);

  // Reads a word and set its index in the vocabulary
  // if the word is not found, set kWordOOV
  // return false iff EOF
  bool ReadWordId(WordIndex* wid);

  int sentence_length_;
  int64_t sentence_id_;
  WordIndex sen_[MAX_SENTENCE_WORDS + 1];
  const Vocabulary& vocab_;
  const WordIndex unk_word_;
  bool reverse_;
  bool done_;
  bool oov_occured_;
};

#endif  // FASTER_RNNLM_WORDS_H_
