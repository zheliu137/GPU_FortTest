
NVIDIA_PATH=/opt/nvidia/hpc_sdk
NV_INSTALL_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/compilers
NV_CUDA_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/cuda/12.3
NV_CUDA_LIB_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/cuda/lib64
NV_CUDA_MATH_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/math_libs/lib64

# NVF90=nvfortran
NVF90=mpifort
IFORT90=mpiifx
CC=mpic++
AR=ar
RANLIB=ranlib

FFILES = mainfort.f90, para.f90
CUFILES = cuda_test.cuf, cuda_test_wrapper.cuf
OBJECTS = mainfort.o para.o 
CUDA_OBJECTS = cuda_test.o cuda_test_wrapper.o
EXENAME = test.x

# NVFLAGS=-Mallocatable=95 -fPIE -gpu=sm_70
NVFLAGS= -g -Mallocatable=95 -fPIE -gpu=sm_86,cuda12.3,nordc,debug -mp=multicore -Minfo=mp

FFLAGS = -qopenmp

NVLIBS=-L$(NV_INSTALL_PATH)/lib -L$(NV_CUDA_PATH) -L$(NV_CUDA_MATH_PATH) -L$(NV_CUDA_LIB_PATH) \
       -lcudafor_113 -lcudafor -lcudadevrt -lcudart -lcublas -lcurand -lcusolver -lcudafor2 -lnvf -lnvomp  -lnvcpumath -lnsnvc -lnvc -lcudaforwrapblas

all: $(OBJECTS) $(CUDA_OBJECTS)
#	    $(NVF90) $(NVFLAGS) $(OBJECTS) $(CUDA_OBJECTS) -o ${EXENAME} ${NVLIBS}
	    $(IFORT90) $(OBJECTS) $(CUDA_OBJECTS) -o ${EXENAME} ${NVLIBS}

# cuda_lib : $(CUDA_OBJECTS)
# 	$(AR) rUuv mycudalib.a $? 
# 	$(RANLIB) mycudalib.a

%.o: %.cuf
	$(NVF90) -c $(NVFLAGS) $< -o $@

# cuda_%.o: cuda_%.cuf
# 	$(NVF90) -c $(NVFLAGS) $< -o $@

%.o: %.f90
#	$(NVF90) -c $(NVFLAGS) $< -o $@
	$(IFORT90) -c $(FFLAGS) $< -o $@

clean:
	    rm -f *.o
	    rm -f *.a
	    rm -f *.mod
	    rm -f ${EXENAME}

cuda_test_wrapper.o : cuda_test.o

cuda_test.o : para.o

mainfort.o : cuda_test_wrapper.o

