NOTHREAD = 0 # set to 1 to disable threads

CC = g++
CFLAGS = -Wall -march=native -funroll-loops -g -D__STDC_FORMAT_MACROS
CFLAGS += -I../ -DEIGEN_DONT_PARALLELIZE # for Eigen
CFLAGS += $(shell $(CC) -dumpversion | awk '{if(NR==1 && $$1>="4.6") print "-Ofast -Wno-unused-result"; else print "-O3";}')
NVCC_CFLAGS = -O3 -march=native -funroll-loops
LDFLAGS = -lm -lrt
ifeq ($(NOTHREAD), 1)
	CFLAGS += -DNOTHREAD
else
	CFLAGS += -pthread
endif

NVCC_RESULT := $(shell which nvcc 2> /dev/null)
NVCC_TEST := $(notdir $(NVCC_RESULT))
ifeq ($(NVCC_TEST), nvcc)
	LDFLAGS += -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas
else
	CFLAGS += -DNOCUDA
endif

OBJ_FILES = rnnlm.o hierarchical_softmax.o nce.o words.o maxent.o nnet.o recurrent.o
ifeq ($(NVCC_TEST), nvcc)
	OBJ_FILES += cuda_softmax.o
endif

all: rnnlm

rnnlm : $(OBJ_FILES)
	$(CC) $^ -o $@ $(CFLAGS) $(LDFLAGS)

hierarchical_softmax.o : hierarchical_softmax.cc hierarchical_softmax.h maxent.h settings.h util.h
	$(CC) $< -c -o $@ $(CFLAGS)

words.o : words.cc words.h settings.h
	$(CC) $< -c -o $@ $(CFLAGS)

rnnlm.o : rnnlm.cc maxent.h nnet.h nce.h settings.h hierarchical_softmax.h words.h recurrent.h util.h program_options.h
	$(CC) $< -c -o $@ $(CFLAGS)

nce.o : nce.cc nce.h cuda_softmax.h maxent.h recurrent.h settings.h words.h util.h
	$(CC) $< -c -o $@ $(CFLAGS)

maxent.o : maxent.cc maxent.h settings.h
	$(CC) $< -c -o $@ $(CFLAGS)

recurrent.o : recurrent.cc recurrent.h settings.h util.h
	$(CC) $< -c -o $@ $(CFLAGS)

nnet.o : nnet.cc nnet.h maxent.h settings.h hierarchical_softmax.h words.h recurrent.h util.h
	$(CC) $< -c -o $@ $(CFLAGS)

cuda_softmax.o : cuda_softmax.cu cuda_softmax.h settings.h
	nvcc $< -c -Xcompiler "$(NVCC_CFLAGS)" -o $@


clean:
	rm -f rnnlm $(OBJ_FILES)
