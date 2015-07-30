#include "cuda_softmax.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cublas_v2.h"

#include "settings.h"


#define MAX_SENTENCE_CHUNK 128

struct CudaStorageInner {
  cublasHandle_t* handle;
};

void AssertCudaSuccess(cudaError_t stat, const char* message) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA ERROR:\n   %s\n   %s\n", message, cudaGetErrorString(stat));
    exit(2);
  }
}

void AssertCudaSuccessLast(const char* message) {
  return AssertCudaSuccess(cudaGetLastError(), message);
}

extern "C"
void InitCudaStorage(CudaStorage* cust, size_t layer_size, size_t vocab_size, size_t maxent_hash_size, Real lnz) {
  cust->layer_size = layer_size;
  cust->vocab_size = vocab_size;
  cust->lnz = lnz;
  cust->maxent_hash_size = maxent_hash_size;

  cudaError_t stat;

  stat = cudaMalloc((void**)&(cust->sm_embedding), layer_size * vocab_size * sizeof(Real));
  AssertCudaSuccess(stat, "Failed to allocate cuda memory for sm_embedding");
  if (maxent_hash_size) {
    stat = cudaMalloc((void**)&(cust->maxent), maxent_hash_size * sizeof(Real));
    AssertCudaSuccess(stat, "Failed to allocate cuda memory for maxent");
    stat = cudaMalloc((void**)&(cust->maxent_indices_all), MAX_SENTENCE_CHUNK * MAX_NGRAM_ORDER * sizeof(uint64_t));
    AssertCudaSuccess(stat, "Failed to allocate cuda memory for maxent_indices_all");
    stat = cudaMalloc((void**)&(cust->maxent_indices_count_all), MAX_SENTENCE_CHUNK * sizeof(int));
    AssertCudaSuccess(stat, "Failed to allocate cuda memory for maxent_indices_count_all");

  } else {
    cust->maxent = 0;
    cust->maxent_indices_all = 0;
    cust->maxent_indices_count_all = 0;
  }

  stat = cudaMalloc((void**)&(cust->hidden_layers), layer_size * MAX_SENTENCE_CHUNK * sizeof(Real));
  AssertCudaSuccess(stat, "Failed to allocate cuda memory for hidden_layers");
  stat = cudaMalloc((void**)&(cust->logprobs), MAX_SENTENCE_CHUNK * sizeof(Real));
  AssertCudaSuccess(stat, "Failed to allocate cuda memory for logprobs");
  stat = cudaMalloc((void**)&(cust->sen_shifted), MAX_SENTENCE_CHUNK * sizeof(WordIndex));
  AssertCudaSuccess(stat, "Failed to allocate cuda memory for sen");

  stat = cudaMalloc((void**)&(cust->scores), vocab_size * MAX_SENTENCE_CHUNK * sizeof(Real));
  AssertCudaSuccess(stat, "Failed to allocate cuda memory for scores");

  stat = cudaMalloc((void**)&(cust->target_scores), MAX_SENTENCE_CHUNK * sizeof(Real));
  AssertCudaSuccess(stat, "Failed to allocate cuda memory for target_scores");
  cust->target_scores_host = new Real[MAX_SENTENCE_CHUNK];

  {
    Real* tmp = new Real[vocab_size];
    for (size_t i = 0; i < vocab_size; ++i)
      tmp[i] = 1;
    stat = cudaMalloc((void**)&(cust->vocab_ones), vocab_size * sizeof(Real));
    AssertCudaSuccess(stat, "Failed to allocate cuda memory for vocab_ones");
    cudaMemcpy(cust->vocab_ones, tmp, vocab_size * sizeof(Real), cudaMemcpyHostToDevice);
    delete[] tmp;
  }

  stat = cudaMalloc((void**)&(cust->sententence_Z), MAX_SENTENCE_CHUNK * sizeof(Real));
  AssertCudaSuccess(stat, "Failed to allocate cuda memory for sententence_Z");
  cust->sententence_Z_host = new Real[MAX_SENTENCE_CHUNK];

  cust->inner = new CudaStorageInner;
  cust->inner->handle = new cublasHandle_t;
  cublasStatus_t cu_stat = cublasCreate(cust->inner->handle);
  if (cu_stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS initialization failed\n");
    exit(1);
  }
}

extern "C"
void FreeCudaStorage(CudaStorage* cust) {
  cudaFree(cust->sm_embedding);
  if (cust->maxent_hash_size != 0) {
    cudaFree(cust->maxent);
    cudaFree(cust->maxent_indices_all);
    cudaFree(cust->maxent_indices_count_all);
  }

  cudaFree(cust->hidden_layers);
  cudaFree(cust->logprobs);
  cudaFree(cust->sen_shifted);

  cudaFree(cust->scores);

  cudaFree(cust->target_scores);
  delete[] cust->target_scores_host;

  cudaFree(cust->vocab_ones);
  cudaFree(cust->sententence_Z);
  delete[] cust->sententence_Z_host;

  cublasDestroy(*cust->inner->handle);
  delete cust->inner->handle;
  delete cust->inner;
}

extern "C"
void UploadNetWeights(CudaStorage* cust, const Real* sm_embedding_cpu, const Real* maxent_cpu) {
  cudaMemcpy(cust->sm_embedding, sm_embedding_cpu, cust->layer_size * cust->vocab_size * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy(cust->maxent, maxent_cpu, cust->maxent_hash_size * sizeof(Real), cudaMemcpyHostToDevice);
}

__global__ void initialize_matrix(Real *a, int rows, int cols, Real value) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x,
        iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < rows && iy < cols) {
        a[ix * cols + iy] = value;
    }
}

__global__ void take_exp(Real *a, int rows, int cols) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x,
        iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < rows && iy < cols) {
        a[ix * cols + iy] = exp(a[ix * cols + iy]);
    }
}

__global__ void add_maxent(int sentence_length, int vocab_size, const WordIndex* sen_shifted, const Real* maxent, const uint64_t* maxent_indices_all, const int* maxent_indices_count_all, Real* scores) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if((ix < sentence_length) && (iy < vocab_size)) {
      int maxent_indices_count = maxent_indices_count_all[ix];
      Real s = scores[ix * vocab_size + iy];;
      for (int i = 0; i < maxent_indices_count; ++i) {
        uint64_t maxent_index = maxent_indices_all[ix * MAX_NGRAM_ORDER + i] + iy;
        s += maxent[maxent_index];
      }
      scores[ix * vocab_size + iy] = exp(s);
    }
}

__global__ void pick_target_scores(const Real* scores, const WordIndex* sen_shifted, size_t vocab_size, Real* output) {
  int word_id = threadIdx.x;
  output[word_id] = scores[word_id * vocab_size + sen_shifted[word_id]];
}


void CublasMultiply_A_BT(cublasHandle_t* handle, float beta, int rows_a, int rows_b, int cols, float* dev_a, float* dev_b, float* dev_c) {
    if (cols == 0)
      return;

    // C <- A * B^T + C * beta
    const float alpha = 1;
    cublasSgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N,
            rows_b, rows_a, cols,
            &alpha,
            dev_b, cols,
            dev_a, cols,
            &beta,
            dev_c, rows_b);
}

extern "C"
void CalculateSoftMax(
    CudaStorage* cust, const Real* hidden_layers,
    const uint64_t* maxent_indices_all, const int* maxent_indices_count_all,
    size_t sentence_length, const WordIndex* sen, Real* logprobs) {
  if (sentence_length > MAX_SENTENCE_CHUNK) {
    CalculateSoftMax(
      cust, hidden_layers,
      maxent_indices_all, maxent_indices_count_all,
      MAX_SENTENCE_CHUNK, sen, logprobs);
    CalculateSoftMax(
      cust, hidden_layers + cust->layer_size * MAX_SENTENCE_CHUNK,
      maxent_indices_all + MAX_SENTENCE_CHUNK * MAX_NGRAM_ORDER, maxent_indices_count_all + MAX_SENTENCE_CHUNK,
      sentence_length - MAX_SENTENCE_CHUNK, sen + MAX_SENTENCE_CHUNK, logprobs + MAX_SENTENCE_CHUNK);
    return;
  }

  cudaError_t stat;

  const size_t layer_size = cust->layer_size;
  const size_t vocab_size = cust->vocab_size;
  stat = cudaMemcpy(cust->hidden_layers, hidden_layers, layer_size * sentence_length * sizeof(Real), cudaMemcpyHostToDevice);
  AssertCudaSuccess(stat, "Failed to copy hidden layers to cuda");
  // copy shifted version of sen to cuda, i.e sen_shifted[i] contains target at position i
  stat = cudaMemcpy(cust->sen_shifted, sen + 1, sentence_length * sizeof(WordIndex), cudaMemcpyHostToDevice);
  AssertCudaSuccess(stat, "Failed to copy sentences to cuda");

  if (cust->maxent_hash_size) {
    stat = cudaMemcpy(cust->maxent_indices_all, maxent_indices_all, sentence_length * MAX_NGRAM_ORDER * sizeof(uint64_t), cudaMemcpyHostToDevice);
    AssertCudaSuccess(stat, "Failed to copy maxent_indices_all to cuda");
    stat = cudaMemcpy(cust->maxent_indices_count_all, maxent_indices_count_all, sentence_length * sizeof(int), cudaMemcpyHostToDevice);
    AssertCudaSuccess(stat, "Failed to copy maxent_indices_count_all to cuda");
  }

  {
    const size_t BLOCK_SIZE = 32;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE),
         blocksPerGrid((sentence_length +  threadsPerBlock.x - 1) / BLOCK_SIZE,
             (vocab_size + threadsPerBlock.y - 1) / BLOCK_SIZE);

    initialize_matrix<<< blocksPerGrid, threadsPerBlock >>>(
        cust->scores, sentence_length, vocab_size, -cust->lnz);

    CublasMultiply_A_BT(
        cust->inner->handle, 1,
        sentence_length, vocab_size, layer_size,
        cust->hidden_layers, cust->sm_embedding, cust->scores);

    if (cust->maxent_hash_size) {
      add_maxent<<< blocksPerGrid, threadsPerBlock >>>(
          sentence_length, vocab_size, cust->sen_shifted,
          cust->maxent, cust->maxent_indices_all, cust->maxent_indices_count_all,
          cust->scores);
    } else {
      take_exp<<< blocksPerGrid, threadsPerBlock >>>(
          cust->scores, sentence_length, vocab_size);
    }
  }

  cudaDeviceSynchronize();

  pick_target_scores<<< 1, sentence_length >>>(
    cust->scores, cust->sen_shifted, vocab_size, cust->target_scores);
  CublasMultiply_A_BT(
      cust->inner->handle, 0,
      sentence_length, 1, vocab_size,
      cust->scores, cust->vocab_ones, cust->sententence_Z);


  stat = cudaMemcpy(cust->target_scores_host, cust->target_scores, sentence_length * sizeof(Real), cudaMemcpyDeviceToHost);
  AssertCudaSuccess(stat, "Failed to copy target scores from cuda");

  cudaMemcpy(cust->sententence_Z_host, cust->sententence_Z, sentence_length * sizeof(Real), cudaMemcpyDeviceToHost);
  AssertCudaSuccess(stat, "Failed to copy probabilities from cuda");

  for (size_t i = 0; i < sentence_length; ++i) {
    Real target_score = cust->target_scores_host[i];
    Real sum = cust->sententence_Z_host[i];
    logprobs[i] = log10(target_score / sum);
  }
}

