#ifndef FASTER_RNNLM_SETTINGS_H_
#define FASTER_RNNLM_SETTINGS_H_

#define MAX_STRING 100            // max length of word in bytes
#define MAX_SENTENCE_WORDS 10000  // max length of a sentence in words
#define MAX_LINE_SIZE 10000000    // max length of a sentence in bytes
#define MAX_NGRAM_ORDER 10        // max direct-order

#define MAX_HSTREE_DEPTH 40       // max supported tree depth
// #define USE_DOUBLE             // uncomment to use double precision

#define RMS_DAMPING_FACTOR 1e-2   // dumping factor for denominator in RMSProp


// #define DETECT_FPE   // if defined the program will fail on FPE; useful to debug nan
// #define PARANOID     // if defined will enforce extra clipping; try this option to wrestle nan

// type of net parameters; all variables that interact with NN should use type Real
#ifdef USE_DOUBLE
typedef double Real;
#else
typedef float Real;
#endif
// type of word indices
typedef unsigned WordIndex;

#ifdef _MSC_VER
#define posix_memalign(where, alignment, nbytes) (*(where) = malloc(nbytes))
#define clock_gettime(b, ts) timespec_get(ts, TIME_UTC)
#endif


#endif  // FASTER_RNNLM_SETTINGS_H_
