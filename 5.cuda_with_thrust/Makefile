
NVIDIA_PATH=/opt/nvidia/hpc_sdk
NV_INSTALL_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/compilers
NV_CUDA_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/cuda/12.3
NV_CUDA_LIB_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/cuda/lib64
NV_CUDA_MATH_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/math_libs/lib64

NVF90=mpifort
NVCC=nvcc
IFORT90=
CC=mpic++
AR=ar
RANLIB=ranlib

# NVFLAGS=-Mallocatable=95 -fPIE -gpu=sm_70
NVFLAGS= -g -Mallocatable=95 -fPIE -gpu=sm_86,cuda12.3,nordc,debug -mp=multicore -Minfo=mp
NVCFLAGS= -g -arch sm_86

FFLAGS=-fPIE -g -traceback -mp=multicore

NVLIBS=-L$(NV_INSTALL_PATH)/lib -L$(NV_CUDA_PATH) -L$(NV_CUDA_MATH_PATH) -L$(NV_CUDA_LIB_PATH) \
       -lcudafor_113 -lcudafor -lcudadevrt -lcudart -lcublas -lcurand -lcusolver -lcudafor2 \
	   -lnvf -lnvomp  -lnvcpumath -lnsnvc -lnvc -lcudaforwrapblas -lstdc++

FFILES = main.cuf
CUFILES = thrust_interf.cuf 
CFILES = thrust_wrapper.cu
OBJECTS = main.o 
CUDA_OBJECTS = thrust_wrapper.o thrust_interf.o 
EXENAME=test.x

all: $(OBJECTS) $(CUDA_OBJECTS)
	    $(NVF90) $(NVFLAGS) $(OBJECTS) $(CUDA_OBJECTS) $(NVLIBS) -o ${EXENAME} -mp=multicore -Minfo=mp -fPIE


# cuda_lib : $(CUDA_OBJECTS)
# 	$(AR) rUuv mycudalib.a $? 
# 	$(RANLIB) mycudalib.a

# cuda_%.o: cuda_%.cuf
%.o: %.cuf
	$(NVF90) -c $(NVFLAGS) $< -o $@

$(OBJECTS): %.o: %.cuf
	$(NVF90) -c $(NVFLAGS) $< -o $@

%.o: %.cu
	$(NVCC) -c $(NVCFLAGS) $< -o $@

clean:
	    rm -f *.o
	    rm -f *.a
	    rm -f *.mod
	    rm -f ${EXENAME}

main.o : thrust_interf.o 
thrust_interf.o : thrust_wrapper.o