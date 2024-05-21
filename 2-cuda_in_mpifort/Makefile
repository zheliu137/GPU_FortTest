
# NVIDIA_PATH=/opt/nvidia/hpc_sdk
NVIDIA_PATH=/cm/shared/apps/spack/0.17.3/gpu/b/opt/spack/linux-rocky8-skylake_avx512/gcc-8.5.0/nvhpc-21.9-4xco23dnhjhodonoufuduj4obaewort7
NV_INSTALL_PATH=$(NVIDIA_PATH)/Linux_x86_64/21.9/compilers
NV_CUDA_PATH=$(NVIDIA_PATH)/Linux_x86_64/21.9/cuda/lib64
NV_CUDA_LIB_PATH=$(NVIDIA_PATH)/Linux_x86_64/21.9/cuda/lib64
NV_CUDA_MATH_PATH=$(NVIDIA_PATH)/Linux_x86_64/21.9/math_libs/lib64

#PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/mpi/openmpi/bin:$PATH

NVF90=nvfortran
IFORT90=mpifort
CC=mpic++
AR=ar
RANLIB=ranlib

FFILES = mainfort.f90, para.f90, cuda_test_wrapper.f90
CUFILES = cuda_test.cuf 
OBJECTS = mainfort.o para.o cuda_test_wrapper.o
CUDA_OBJECTS = cuda_test.o 
EXENAME=test.x

# NVFLAGS=-Mallocatable=95 -fPIE -gpu=sm_70
# NVFLAGS= -g -Mallocatable=95 -fPIE -gpu=sm_86,cuda12.3,nordc,debug
NVFLAGS= -g -Mallocatable=95 -fPIE -gpu=cc70,nordc,debug

FFLAGS=-fPIE -g -traceback

NVLIBS=-L$(NV_INSTALL_PATH)/lib -L$(NV_CUDA_PATH) -L$(NV_CUDA_MATH_PATH) -L$(NV_CUDA_LIB_PATH) \
       -lcudafor_114 -lcudafor -lcudadevrt -lcudart -lcublas -lcurand -lcusolver -lcudafor2 -lnvf -lnvomp  -lnvcpumath -lnsnvc -lnvc -lcudaforwrapblas

all: $(OBJECTS) cuda_lib
	    $(IFORT90) $(FFLAGS) $(OBJECTS) mycudalib.a $(NVLIBS) -o ${EXENAME}

cuda_test_wrapper.o : cuda_test.o

cuda_lib : $(CUDA_OBJECTS)
	$(AR) rUuv mycudalib.a $? 
	$(RANLIB) mycudalib.a

# $(CUDA_OBJECTS): cuda_%.o: cuda_%.cuf
cuda_%.o: cuda_%.cuf
	$(NVF90) -c $(NVFLAGS) $< -o $@

$(OBJECTS): %.o: %.f90
	$(IFORT90) -c $(FFLAGS) $< -o $@

cuda_test.o : para.o

clean:
	    rm -f *.o
	    rm -f *.a
	    rm -f *.mod
	    rm -f ${EXENAME}

cuda_test.o : para.o

mainfort.o : cuda_test_wrapper.o