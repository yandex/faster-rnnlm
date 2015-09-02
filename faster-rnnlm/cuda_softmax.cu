#include "cuda_softmax.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cublas_v2.h"

#include "settings.h"


#define MAX_SENTENCE_CHUNK 128

struct CudaStorageInner {
  cublasHandle_t* handle;
  cudaStream_t stream_data;
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

void InitCudaStorage(CudaStorage* cust, size_t layer_size, size_t vocab_size, size_t maxent_hash_size, Real lnz, bool memory_efficient_maxent) {
  cust->layer_size = layer_size;
  cust->vocab_size = vocab_size;
  cust->lnz = lnz;
  cust->memory_efficient_maxent = memory_efficient_maxent;
  cust->maxent_hash_size = maxent_hash_size;

  cudaError_t stat;

  stat = cudaMalloc((void**)&(cust->sm_embedding), layer_size * vocab_size * sizeof(Real));
  AssertCudaSuccess(stat, "Failed to allocate cuda memory for sm_embedding");
  if (maxent_hash_size) {
    if (memory_efficient_maxent) {
      cust->maxent_indices_all = NULL;
      cust->maxent_cpu = new Real[maxent_hash_size];

      stat = cudaMalloc((void**)&(cust->maxent), MAX_NGRAM_ORDER * vocab_size * MAX_SENTENCE_CHUNK * sizeof(Real));
      AssertCudaSuccess(stat, "Failed to allocate cuda memory for maxent");
    } else {
      cust->maxent_cpu = NULL;

      stat = cudaMalloc((void**)&(cust->maxent), maxent_hash_size * sizeof(Real));
      AssertCudaSuccess(stat, "Failed to allocate cuda memory for maxent");
      stat = cudaMalloc((void**)&(cust->maxent_indices_all), MAX_SENTENCE_CHUNK * MAX_NGRAM_ORDER * sizeof(uint64_t));
      AssertCudaSuccess(stat, "Failed to allocate cuda memory for maxent_indices_all");
    }

    stat = cudaMalloc((void**)&(cust->maxent_indices_count_all), MAX_SENTENCE_CHUNK * sizeof(int));
    AssertCudaSuccess(stat, "Failed to allocate cuda memory for maxent_indices_count_all");

  } else {
    cust->maxent = NULL;
    cust->maxent_indices_all = NULL;
    cust->maxent_indices_count_all = NULL;
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
    for (size_t i = 0; i < vocab_size; ++i) {
      tmp[i] = 1;
    }
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

  stat = cudaStreamCreate(&cust->inner->stream_data);
  AssertCudaSuccess(stat, "Failed to create stream");
}

void FreeCudaStorage(CudaStorage* cust) {
  cudaFree(cust->sm_embedding);
  if (cust->maxent_hash_size != 0) {
    delete[] cust->maxent_cpu;
    cudaFree(cust->maxent);
    if (!cust->memory_efficient_maxent) {
      cudaFree(cust->maxent_indices_all);
    }
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
  cudaStreamDestroy(cust->inner->stream_data);
  delete cust->inner;
}

void UploadNetWeights(CudaStorage* cust, const Real* sm_embedding_cpu, const Real* maxent_cpu) {
  cudaMemcpy(cust->sm_embedding, sm_embedding_cpu, cust->layer_size * cust->vocab_size * sizeof(Real), cudaMemcpyHostToDevice);
  if (cust->memory_efficient_maxent) {
    for (size_t i = 0; i < cust->maxent_hash_size; ++i) {
      cust->maxent_cpu[i] = maxent_cpu[i];
    }
  } else {
    cudaMemcpy(cust->maxent, maxent_cpu, cust->maxent_hash_size * sizeof(Real), cudaMemcpyHostToDevice);
  }
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
    if ((ix < sentence_length) && (iy < vocab_size)) {
      int maxent_indices_count = maxent_indices_count_all[ix];
      Real s = scores[ix * vocab_size + iy];
      for (int i = 0; i < maxent_indices_count; ++i) {
        uint64_t maxent_index = maxent_indices_all[ix * MAX_NGRAM_ORDER + i] + iy;
        s += maxent[maxent_index];
      }
      scores[ix * vocab_size + iy] = exp(s);
    }
}

__global__ void add_prepared_maxent(int sentence_length, int vocab_size, const WordIndex* sen_shifted, const Real* maxent_prepared, const int* maxent_indices_count_all, int max_maxent_order, Real* scores) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if((ix < sentence_length) && (iy < vocab_size)) {
      int maxent_indices_count = maxent_indices_count_all[ix];
      Real s = scores[ix * vocab_size + iy];
      for (int i = 0; i < maxent_indices_count; ++i) {
        uint64_t maxent_index = (max_maxent_order * ix + i) * vocab_size + iy;
        s += maxent_prepared[maxent_index];
      }
      scores[ix * vocab_size + iy] = exp(s);
    }
}

__global__ void pick_target_scores(const Real* scores, const WordIndex* sen_shifted, size_t vocab_size, Real* output) {
  int word_id = threadIdx.x;
  output[word_id] = scores[word_id * vocab_size + sen_shifted[word_id]];
}


void CublasMultiply_A_BT(cublasHandle_t* handle, Real beta, int rows_a, int rows_b, int cols, Real* dev_a, Real* dev_b, Real* dev_c) {
    if (cols == 0)
      return;

    // C <- A * B^T + C * beta
    const Real alpha = 1;
#ifdef USE_DOUBLE
    cublasDgemm
#else
    cublasSgemm
#endif
      (*handle, CUBLAS_OP_T, CUBLAS_OP_N,
            rows_b, rows_a, cols,
            &alpha,
            dev_b, cols,
            dev_a, cols,
            &beta,
            dev_c, rows_b);
}

void CalculateSoftMax(
    CudaStorage* cust, const Real* hidden_layers,
    const uint64_t* maxent_indices_all, const int* maxent_indices_count_all,
    size_t sentence_length, const WordIndex* sen, Real* logprobs) {
  if (sentence_length > MAX_SENTENCE_CHUNK) {
    // process long sentences by chunks
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
  // copy shifted version of sen to cuda, i.e sen_shifted[i] contains target at position i,
  // sen[i] contains input at position i
  stat = cudaMemcpy(cust->sen_shifted, sen + 1, sentence_length * sizeof(WordIndex), cudaMemcpyHostToDevice);
  AssertCudaSuccess(stat, "Failed to copy sentences to cuda");

  if (cust->maxent_hash_size) {
    if (!cust->memory_efficient_maxent) {
      stat = cudaMemcpyAsync(cust->maxent_indices_all, maxent_indices_all, sentence_length * MAX_NGRAM_ORDER * sizeof(uint64_t), cudaMemcpyHostToDevice, cust->inner->stream_data);
      AssertCudaSuccess(stat, "Failed to copy maxent_indices_all to cuda");
    }
    stat = cudaMemcpyAsync(cust->maxent_indices_count_all, maxent_indices_count_all, sentence_length * sizeof(int), cudaMemcpyHostToDevice, cust->inner->stream_data);
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
      if (cust->memory_efficient_maxent) {
        int max_maxent_order = 0;
        for (int i = 0; i < sentence_length; ++i) {
          int maxent_order = maxent_indices_count_all[i];
          max_maxent_order = (max_maxent_order < maxent_order) ? maxent_order : max_maxent_order;
        }

        for (int pos = 0; pos < sentence_length; ++pos) {
          for (int i = 0; i < maxent_indices_count_all[pos]; ++i) {
            uint64_t maxent_index = maxent_indices_all[pos * MAX_NGRAM_ORDER + i];
            Real* dst = cust->maxent + (max_maxent_order * pos + i) * vocab_size;
            Real* src = cust->maxent_cpu + maxent_index;
            cudaMemcpyAsync(dst, src, vocab_size * sizeof(Real), cudaMemcpyHostToDevice, cust->inner->stream_data);
          }
        }

        cudaStreamSynchronize(cust->inner->stream_data);
        add_prepared_maxent<<< blocksPerGrid, threadsPerBlock >>>(
            sentence_length, vocab_size, cust->sen_shifted,
            cust->maxent, cust->maxent_indices_count_all, max_maxent_order,
            cust->scores);
      } else {
        cudaStreamSynchronize(cust->inner->stream_data);
        add_maxent<<< blocksPerGrid, threadsPerBlock >>>(
            sentence_length, vocab_size, cust->sen_shifted,
            cust->maxent, cust->maxent_indices_all, cust->maxent_indices_count_all,
            cust->scores);
      }
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

