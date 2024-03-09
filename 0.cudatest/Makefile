NVIDIA_PATH=/opt/nvidia/hpc_sdk
NV_INSTALL_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/compilers
NV_CUDA_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/cuda/12.3
NV_CUDA_MATH_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/math_libs/lib64

NVF90=nvfortran

NVFLAGS=-Mallocatable=95 -fPIE
FFLAGS=-fPIE 


NVLIBS=-cudalib=cublas,cusolver,curand

all: kernels libs

kernels: test_kernels.o
	$(NVF90) test_kernels.cuf -o test_kernels.x
	./test_kernels.x

libs: test_libs_init test_cublas_blas

test_cublas_blas: test_cublas_blas.o
	$(NVF90) -lblas $(NVLIBS) test_cublas_blas.cuf -o test_cublas_blas.x
	./test_cublas_blas.x

test_libs_init: test_libs_init.o
	$(NVF90) $(NVLIBS) test_libs_init.cuf -o test_libs_init.x
	./test_libs_init.x

test_%.o: test_%.cuf
	$(NVF90) -c $(NVFLAGS) $< -o $@

clean:
	    rm -f *.o
	    rm -f *.mod
	    rm -f *.x