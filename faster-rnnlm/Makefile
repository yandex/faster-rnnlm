NOTHREAD = 0 # set to 1 to disable threads

CC = g++
CFLAGS = -Wall -march=native -funroll-loops -g -D__STDC_FORMAT_MACROS
CFLAGS += -I../ -DEIGEN_DONT_PARALLELIZE # for Eigen
CFLAGS += $(shell $(CC) -dumpversion | awk '{if(NR==1 && $$1>="4.6") print "-Ofast"; else print "-O3";}')
NVCC_CFLAGS = -O3 -march=native -funroll-loops
LDFLAGS = -lm -lstdc++
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

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
	CFLAGS += -DNORT
else
	LDFLAGS += -lrt
endif

LAYER_OBJ_FILES = layers/simple_layer.o layers/gru_layer.o layers/scrn_layer.o layers/layer_stack.o layers/interface.o
OBJ_FILES = hierarchical_softmax.o nce.o words.o maxent.o nnet.o $(LAYER_OBJ_FILES)
ifeq ($(NVCC_TEST), nvcc)
	OBJ_FILES += cuda_softmax.o
endif

all: rnnlm

rnnlm : rnnlm.o $(OBJ_FILES)
	$(CC) $^ -o $@ $(CFLAGS) $(LDFLAGS)

hierarchical_softmax.o : hierarchical_softmax.cc hierarchical_softmax.h maxent.h settings.h util.h
	$(CC) $< -c -o $@ $(CFLAGS)

words.o : words.cc words.h settings.h
	$(CC) $< -c -o $@ $(CFLAGS)

rnnlm.o : rnnlm.cc maxent.h nnet.h nce.h settings.h hierarchical_softmax.h words.h layers/interface.h util.h program_options.h
	$(CC) $< -c -o $@ $(CFLAGS)

nce.o : nce.cc nce.h cuda_softmax.h maxent.h layers/interface.h settings.h words.h util.h
	$(CC) $< -c -o $@ $(CFLAGS)

maxent.o : maxent.cc maxent.h util.h settings.h
	$(CC) $< -c -o $@ $(CFLAGS)

nnet.o : nnet.cc nnet.h maxent.h settings.h hierarchical_softmax.h words.h layers/interface.h util.h
	$(CC) $< -c -o $@ $(CFLAGS)

cuda_softmax.o : cuda_softmax.cu cuda_softmax.h settings.h
	nvcc $< -c -Xcompiler "$(NVCC_CFLAGS)" -o $@

layers/simple_layer.o : layers/simple_layer.cc layers/simple_layer.h layers/interface.h layers/activation_functions.h layers/util.h
	$(CC) $< -c -o $@ $(CFLAGS)

layers/scrn_layer.o : layers/scrn_layer.cc layers/scrn_layer.h layers/interface.h layers/util.h layers/activation_functions.h settings.h util.h
	$(CC) $< -c -o $@ $(CFLAGS)

layers/gru_layer.o : layers/gru_layer.cc layers/gru_layer.h layers/interface.h layers/util.h layers/activation_functions.h settings.h util.h
	$(CC) $< -c -o $@ $(CFLAGS)

layers/layer_stack.o : layers/layer_stack.cc layers/layer_stack.h layers/interface.h layers/util.h settings.h util.h
	$(CC) $< -c -o $@ $(CFLAGS)

layers/interface.o : layers/interface.cc layers/interface.h layers/util.h settings.h util.h $(LAYER_OBJ_FILES:.o=.h)
	$(CC) $< -c -o $@ $(CFLAGS)

test_gradients/test_gradients : test_gradients/test_gradients.cc $(OBJ_FILES)
	$(CC) $^ -o $@ -std=gnu++0x $(CFLAGS) $(LDFLAGS)

clean:
	rm -f rnnlm $(OBJ_FILES)
