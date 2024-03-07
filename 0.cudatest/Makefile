
#PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/mpi/openmpi/bin:$PATH
NVIDIA_PATH=/opt/nvidia/hpc_sdk
NV_INSTALL_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/compilers
NV_CUDA_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/cuda/12.3
NV_CUDA_MATH_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/math_libs/lib64

NVF90=nvfortran

NVFLAGS=-Mallocatable=95 -fPIE
FFLAGS=-fPIE 

# NVLIBS=-L$(NV_INSTALL_PATH)/lib -L$(NV_CUDA_PATH) -L$(NV_CUDA_MATH_PATH) -lcudafor_113 -lcudafor -lcudadevrt -lcudart -lcublas -lcurand -lcusolver -lcudafor2 -lnvf -lnvomp  -lnvcpumath -lnsnvc -lnvc
# NVLIBS=-L$(NV_INSTALL_PATH)/lib -L$(NV_CUDA_PATH) -L$(NV_CUDA_MATH_PATH) \
    #    -lcudafor_113 -lcudafor -lcudadevrt -lcudart -lcublas -lcurand -lcusolver -lcudafor2 -lnvf -lnvomp  -lnvcpumath -lnsnvc -lnvc -lcudaforwrapblas
# NVLIBS=-lblas -cudalib=cublas,cusolver,curand
NVLIBS=-cudalib=cublas,cusolver,curand
    #    -lcudafor_113 -lcudafor -lcudadevrt -lcudart -lcublas -lcurand -lcusolver -lcudafor2 -lnvf -lnvomp  -lnvcpumath -lnsnvc -lnvc

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

# nvfortran -c test.o test.cuf -lblas -cudalib=cublas,cusolver,curand

# nvfortran -fPIE -c test.o test.cuf
# nvfortran -fPIE -c test_wrapper.o test_wrapper.f90
# nvfortran -fPIE -c mainfort.o mainfort.f90

# ar ruv libtest.a test.o test_wrapper.o

# nvfortran -fPIE -o a.out -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver libtest.a mainfort.o
# nvfortran -fPIE -o a.out -cudalib=cublas,cusolver,curand libtest.a mainfort.o test.o test_wrapper.o
# ifx -o a.out $(NVLIBS) -static-libstdc++ -cxxlib libtest.a mainfort.o

# ifx -o a.out -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver -static-libstdc++ -cxxlib main.o test.o test_wrapper.o
# nvfortran -o a.out main.o test.o test_wrapper.o -cudalib=cublas,cusolver,curand
# ifx -c main.f90 -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver -static-libstdc++ -cxxlib
# ifx -static-intel -o a.out main.f90 lib.a -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver -static-libstdc++ -cxxlib

# ./a.out