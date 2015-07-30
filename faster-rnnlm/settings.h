#ifndef FASTER_RNNLM_SETTINGS_H_
#define FASTER_RNNLM_SETTINGS_H_

#define GRAD_CLIPPING 1

#define MAX_STRING 100            // max length of word in bytes
#define MAX_SENTENCE_WORDS 10000  // max length of a sentence in words
#define MAX_LINE_SIZE 10000000    // max length of a sentence in bytes
#define MAX_NGRAM_ORDER 10        // max direct-order

#define ARITY 2                   // arity of HS-tree
#define MAX_HSTREE_DEPTH 40       // max supported tree depth


// #define DETECT_FPE   // if defined the program will fail on FPE; useful to debug nan
// #define PARANOID     // if defined will enforce extra clipping; try this option to wrestle nan

// type of net parameters; all variables that interact with NN should use type Real
typedef float Real;
// type of word indices
typedef unsigned WordIndex;


#endif  // FASTER_RNNLM_SETTINGS_H_
