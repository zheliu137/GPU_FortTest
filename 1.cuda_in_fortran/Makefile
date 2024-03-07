
#PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/mpi/openmpi/bin:$PATH
NVIDIA_PATH=/opt/nvidia/hpc_sdk
NV_INSTALL_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/compilers
NV_CUDA_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/cuda/12.3
NV_CUDA_LIB_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/cuda/lib64
NV_CUDA_MATH_PATH=$(NVIDIA_PATH)/Linux_x86_64/24.1/math_libs/lib64

NVF90=nvfortran
IFORT90=mpifort
CC=mpic++
AR=ar
RANLIB=ranlib

FFILES = mainfort.f90, para.f90
CUFILES = cuda_test.cuf cuda_test_wrapper.cuf
OBJECTS = mainfort.o para.o
CUDA_OBJECTS = cuda_test.o cuda_test_wrapper.o
EXENAME=text.x

# NVFLAGS=-Mallocatable=95 -fPIE -gpu=sm_70
NVFLAGS= -g -Mallocatable=95 -fPIE -gpu=sm_86,cuda12.3,nordc,debug

# NVFLAGS=-fPIE
# NVFLAGS=-fPIE
FFLAGS=-fPIE -g -traceback

# NVLIBS=-L$(NV_INSTALL_PATH)/lib -L$(NV_CUDA_PATH) -L$(NV_CUDA_MATH_PATH) -lcudafor_113 -lcudafor -lcudadevrt -lcudart -lcublas -lcurand -lcusolver -lcudafor2 -lnvf -lnvomp  -lnvcpumath -lnsnvc -lnvc
NVLIBS=-L$(NV_INSTALL_PATH)/lib -L$(NV_CUDA_PATH) -L$(NV_CUDA_MATH_PATH) -L$(NV_CUDA_LIB_PATH) \
       -lcudafor_113 -lcudafor -lcudadevrt -lcudart -lcublas -lcurand -lcusolver -lcudafor2 -lnvf -lnvomp  -lnvcpumath -lnsnvc -lnvc -lcudaforwrapblas
    #    -lcudafor_113 -lcudafor -lcudadevrt -lcudart -lcublas -lcurand -lcusolver -lcudafor2 -lnvf -lnvomp  -lnvcpumath -lnsnvc -lnvc

# all: $(OBJECTS) $(CUDA_OBJECTS)
all: $(OBJECTS) cuda_lib
	    $(IFORT90) -c $(FFLAGS) $(FFILES)
	    $(NVF90) $(NVFLAGS) -c $(CUFILES)
#	    $(IFORT90) $(FFLAGS) $(OBJECTS) $(CUDA_OBJECTS) $(NVLIBS) -o ${EXENAME}
	    $(IFORT90) $(FFLAGS) $(OBJECTS) mycudalib.a $(NVLIBS) -o ${EXENAME}

test_wrapper.o : cuda_test.o

cuda_lib : $(CUDA_OBJECTS)
	$(AR) rUuv mycudalib.a $? 
	$(RANLIB) mycudalib.a

# $(CUDA_OBJECTS): cuda_%.o: cuda_%.cuf
cuda_%.o: cuda_%.cuf
	$(NVF90) -c $(NVFLAGS) $< -o $@

$(OBJECTS): %.o: %.f90
	$(IFORT90) -c $(FFLAGS) $< -o $@

cuda_test.o : para.o


# %.cuf: %.o

clean:
	    rm -f *.o
	    rm -f *.mod
	    rm -f ${EXENAME}

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