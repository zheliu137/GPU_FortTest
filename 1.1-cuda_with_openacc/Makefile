NVIDIA_PATH=/opt/nvidia/hpc_sdk
NV_INSTALL_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/compilers
NV_CUDA_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/cuda/12.3
NV_CUDA_LIB_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/cuda/lib64
NV_CUDA_MATH_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/math_libs/lib64

NVF90=nvfortran
IFORT90=nvfortran
CC=mpicc
AR=ar
RANLIB=ranlib

FFILES = mainfort.f90 para.f90 cuda_test_wrapper.f90
CUFILES = cuda_test.cuf 
OBJECTS = mainfort.o para.o cuda_test_wrapper.o
CUDA_OBJECTS = cuda_test.o 
EXENAME=test.x

# NVFLAGS=-Mallocatable=95 -fPIE -gpu=sm_70
NVFLAGS=-g -Minfo -Mallocatable=95 -fPIE -gpu=sm_86,cuda12.3,nordc,debug -acc=gpu -cuda

# NVFLAGS=-fPIE
FFLAGS=-fPIE -g -traceback

NVLIBS=-cudalib=cufft,cublas,cusolver,curand

# NVLIBS=-L$(NV_INSTALL_PATH)/lib -L$(NV_CUDA_PATH) -L$(NV_CUDA_MATH_PATH) -L$(NV_CUDA_LIB_PATH) \
#        -lcudafor_113 -lcudafor -lcudadevrt -lcudart -lcublas -lcurand -lcusolver -lcudafor2 -lnvf -lnvomp  -lnvcpumath -lnsnvc -lnvc -lcudaforwrapblas

all: $(OBJECTS) cuda_lib
	    $(IFORT90) $(NVFLAGS) $(OBJECTS) mycudalib.a $(NVLIBS) -o ${EXENAME}

cuda_test_wrapper.o : cuda_test.o

cuda_lib : $(CUDA_OBJECTS)
	$(AR) rUuv mycudalib.a $? 
	$(RANLIB) mycudalib.a

# $(CUDA_OBJECTS): cuda_%.o: cuda_%.cuf
cuda_%.o: cuda_%.cuf
	$(NVF90) -c $(NVFLAGS) $< -o $@

$(OBJECTS): %.o: %.f90
	$(IFORT90) -c $(NVFLAGS) $< -o $@

cuda_test.o : para.o

mainfort.o : cuda_test_wrapper.o

clean:
	    rm -f *.o
	    rm -f *.mod
		rm -f *.a
	    rm -f ${EXENAME}
