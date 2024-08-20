#compilers
CC=/usr/local/cuda-12.5/bin/nvcc

NVCC_FLAGS = -O3 -ccbin /usr/bin/gcc -m64 -gencode arch=compute_80,code=sm_80

# #ENVIRONMENT_PARAMETERS
# CUDA_INSTALL_PATH = /usr/local/cuda-12.0

CUDA_LIBS = -lcusparse -lcublas
LIBS =  -lineinfo $(CUDA_LIBS)

#options
OPTIONS = -Xcompiler -fopenmp-simd

double:
	$(CC) $(NVCC_FLAGS) src/main_f64.cu -o spmv_double  -D f64 $(OPTIONS) $(LIBS)

nec:
	$(CC) $(NVCC_FLAGS) src/necSpMV_fp64.cu -o nec -D f64 $(OPTIONS) $(LIBS) 

half:
	$(CC) $(NVCC_FLAGS) src/main_f16.cu -o spmv_half $(OPTIONS) $(LIBS) 

analyze:
	$(CC) $(NVCC_FLAGS) src/main_SF.cu -o analize $(OPTIONS)



clean:
	rm -rf spmv_double
	rm -rf spmv_half
	rm -rf nec
	# rm data/*.csv