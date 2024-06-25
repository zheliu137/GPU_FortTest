MODULE cudamod
  USE cudafor
  USE cublas
  USE cusolverDn
  USE curand
  USE openacc
  use omp_lib
  USE parameters
  IMPLICIT NONE

CONTAINS

  ATTRIBUTES(GLOBAL) SUBROUTINE ArrayAdd( A, B, C, length, nloop )
      integer :: idx
      REAL(kind=DP),DEVICE :: A(*), B(*), C(*)
      integer, value :: length, nloop
      integer :: iloop
      REAL(kind=DP), shared :: AA(32)
      idx = blockIdx%x * blockDim%x + threadIdx%x + 1
      if( idx < length ) then
        AA(threadIdx%x+1)=0.0
        do iloop=1,nloop
          AA(threadIdx%x+1) = AA(threadIdx%x+1) + B(idx) + C(idx)
        ENDDO
        A(idx)=AA(threadIdx%x+1)
      endif
  END SUBROUTINE ArrayAdd
  Attributes(global) SUBROUTINE kernel_print(a, b)
    REAL(kind=DP) :: a(*), b(*)
    INTEGER :: i
    i = blockDim%x*(blockIdx%x - 1) + threadIdx%x
    ! j = bitrev8(threadIdx%x-1) + 1
    ! print*, " thread : ", threadIdx%x, threadIdx%y, " DP = ", DP
    ! write(*,"(' thread : ', 2I6, ' a(', I6 ') = ', F15.5, ' b(', I6 ') = ', F15.5 )") &
    !                  threadIdx%x, threadIdx%y,  i, a(i), i, b(i)
    ! write(*,*) ' thread : ', threadIdx%x, threadIdx%y, "a(" , i, ') = ',  a(i)
    PRINT *, ' thread : ', threadIdx%x, threadIdx%y, i, a(i)
  END SUBROUTINE kernel_print

  SUBROUTINE testcudalib
    !
    IMPLICIT NONE
    INTEGER :: istat
    TYPE(curandGenerator) :: gene_rand
    TYPE(cublasHandle) :: h_blas
    TYPE(cusolverDnHandle) :: h_solver

    REAL(kind=DP), ALLOCATABLE, device :: d_a(:)
    ! real(kind=DP),allocatable        :: h_a(:)
    REAL(kind=DP)        :: h_a(10)
    REAL(kind=DP)        :: h_b(10)
    TYPE(cudaChannelFormatDesc) :: desc

    INTEGER :: i
    PRINT *, "DP = ", DP
    DO i = 1, 10
      h_a(i) = i
    END DO
    istat = cudaMalloc(d_a, 10)
    istat = cudaMemcpy(d_a, h_a, 10)
    istat = cudaMemcpy(h_b, d_a, 10)

    PRINT *, "h_a = ", h_a
    PRINT *, "h_b = ", h_b
    ! print*,"d_a = ", d_a

    istat = cublasCreate(h_blas)
    PRINT *, "cublasCreate returned ", istat
    istat = cuSolverDnCreate(h_solver)
    PRINT *, "cuSolverDnCreate returned ", istat
    istat = curandCreateGenerator(gene_rand, CURAND_RNG_PSEUDO_XORWOW)
    PRINT *, "curandCreateGenerator returned ", istat

    istat = cublasDestroy(h_blas)
    PRINT *, "cublasDestroy returned ", istat
    istat = cuSolverDnDestroy(h_solver)
    PRINT *, "cusolverDnDestroy returned ", istat
    istat = curandDestroyGenerator(gene_rand)
    PRINT *, "curandDestroyGenerator returned ", istat

    istat = cudaFree(d_a)
    PRINT *, "cudaFree returned ", istat

  END SUBROUTINE testcudalib

  SUBROUTINE testcudaker
    !
    IMPLICIT NONE

    REAL(kind=DP), ALLOCATABLE, device :: d_a(:), d_b(:)
    ! real(kind=DP),allocatable        :: h_a(:)
    REAL(kind=DP)        :: h_a(200)
    REAL(kind=DP)        :: h_b(200)
    TYPE(cudaChannelFormatDesc) :: desc

    INTEGER :: i
    INTEGER :: istat
    TYPE(dim3) :: dimGrid, dimBlock

    DO i = 1, 200
      h_a(i) = i
    END DO
    istat = cudaMalloc(d_a, 200)
    istat = cudaMalloc(d_b, 200)

    dimGrid = dim3(1, 1, 1)
    dimBlock = dim3(2, 2, 1)
    CALL kernel_print <<< dimGrid, dimBlock >>> (d_a, d_b)

    istat = cudaDeviceSynchronize()

  END SUBROUTINE testcudaker

  SUBROUTINE testopenacc
    !
    IMPLICIT NONE
    !
    INTEGER :: n, nloop
    REAL, ALLOCATABLE :: a(:, :), b(:, :), c(:, :)
    REAL, ALLOCATABLE :: a2(:), b2(:), c2(:)
    REAL :: t1, t2, t3, t4
    INTEGER :: i, j, k, iloop
    INTEGER :: max_idx(2)
    !
    REAL(KIND=DP), DEVICE, ALLOCATABLE :: d_a(:), d_b(:), d_c(:)
    ! real(kind=DP),allocatable        :: h_a(:)
    REAL(kind=DP)        :: h_a(200)
    REAL(kind=DP)        :: h_b(200)
    TYPE(cudaChannelFormatDesc) :: desc
    INTEGER :: istat
    TYPE(dim3) :: dimGrid, dimBlock

    type (cudaEvent) :: startEvent, stopEvent, dummyEvent
    REAL :: time

    istat = cudaEventCreate(startEvent)
    istat = cudaEventCreate(stopEvent)  
    istat = cudaEventCreate(dummyEvent)  
    
    n = 200
    nloop = 200
    PRINT*, " Calculate sin(x)^2 + cos(x)^2 with x being every element in ", n, " by ", n, " array ", nloop, " times."
    !
    ALLOCATE (a(n, n), b(n, n), c(n, n))
    CALL RANDOM_SEED()
    CALL RANDOM_NUMBER(c)
    a=0.0
    b=0.0
    !call acc_init( acc_device_nvidia )
    !!$acc data copy(a(:,:), c(:,:))
    CALL CPU_TIME(t1)
    !$ACC kernels loop
    DO i = 1, n
      DO j = 1, n
        DO iloop = 1, nloop
          a(i, j) = a(i, j) + SIN(c(i, j)) ** 2 + COS(c(i, j)) ** 2
        END DO
      END DO
    END DO
    CALL CPU_TIME(t2)
    !!$acc end data
    PRINT*, t2-t1, " secs by OpenACC on GPU"
    !
    CALL CPU_TIME(t2)
    DO i = 1, n
      DO j = 1, n
        DO iloop = 1, nloop
          b(i, j) = b(i, j) + SIN(c(i, j))**2 + COS(c(i, j))**2
        END DO
      END DO
    END DO
    CALL CPU_TIME(t3)
    
    PRINT*, t3-t2, " secs on CPU"

    IF (MAXVAL(ABS(a - b)) > 1E-5*nloop) THEN
      max_idx(:)=MAXLOC(ABS(a - b))
      PRINT *, "ERROR!!! CPU and GPU got different results."
      PRINT *, MAXVAL(a), MINVAL(a), MAXVAL(b), MINVAL(b), MAXVAL(ABS(a - b)), max_idx(1), max_idx(2)
      PRINT *, a(max_idx(1), max_idx(2)), b(max_idx(1), max_idx(2))
      PRINT *, c(max_idx(1), max_idx(2)), nloop*(SIN(c(max_idx(1), max_idx(2)))**2 + COS(c(max_idx(1), max_idx(2)))**2)
    END IF
    
    DEALLOCATE(a,b,c)
    
    n=n**2
    PRINT*, " Add b and c to a ", nloop, " times. ", " They are arrays with length ", n

    ALLOCATE (a2(n), b2(n), c2(n))
    a2=0.0
    CALL RANDOM_NUMBER(b2)
    CALL RANDOM_NUMBER(c2)
    !$acc data copy(a2(:), b2(:), c2(:))
    !CALL acc_wait_all()
    CALL CPU_TIME(t1)
    !$ACC kernels loop
    DO i = 1, n
      DO iloop = 1, nloop
        a2(i) = a2(i) + b2(i) + c2(i)
      END DO
    END DO
    !CALL acc_wait_all()
    CALL CPU_TIME(t2)
    !$acc end data
    
    PRINT*, t2-t1, " secs on OpenACC"
    
    CALL CPU_TIME(t2)
    DO i = 1, n
      DO iloop = 1, nloop
        a2(i) = a2(i) + b2(i) + c2(i)
      END DO
    END DO

    CALL CPU_TIME(t3)
    PRINT*, t3-t2, " secs on CPU"
    
    ALLOCATE(d_a(n), d_b(n), d_c(n))
    d_a=0.0
    d_b(:)=b2(:)
    d_c(:)=c2(:)
    CALL CPU_TIME(t2)

    dimBlock = 32
    dimGrid = (n+dimBlock%x-1)/dimBlock%x

    istat = cudaEventRecord(startEvent,0)
    call ArrayAdd<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, n, nloop)
    istat = cudaEventRecord(stopEvent,0)
    istat = cudaEventSynchronize(stopEvent)
    istat = cudaEventElapsedTime(time, startEvent, stopEvent)

    CALL CPU_TIME(t4)

    PRINT*, t4-t2, " secs on GPU kernel by CPU timers excluding allocations"
    PRINT*, t4-t3, " secs on GPU kernel by CPU timers including allocations"
    PRINT*, time/1000, " secs on GPU kernel by GPU timer"

    DEALLOCATE(a2, b2, c2)
    DEALLOCATE(d_a, d_b, d_c)

  END SUBROUTINE testopenacc

  SUBROUTINE openacc_multipurpose
    !
    IMPLICIT NONE
    !
    INTEGER :: n, nloop
    REAL, ALLOCATABLE :: a(:), b(:), c(:)
    REAL, ALLOCATABLE :: a2(:), b2(:), c2(:)
    ! REAL, DEVICE, ALLOCATABLE :: da(:), db(:), dc(:)
    REAL :: t0, t1, t2, t3, t4, t5
    REAL :: t_omp(10)
    INTEGER :: i, j, k, iloop
    INTEGER :: max_idx(2)
    !
    REAL(kind=DP)        :: h_a(200)
    REAL(kind=DP)        :: h_b(200)
    TYPE(cudaChannelFormatDesc) :: desc
    INTEGER :: istat
    TYPE(dim3) :: dimGrid, dimBlock

    type (cudaEvent) :: startEvent, stopEvent, dummyEvent
    REAL :: time
    integer(kind=acc_device_kind) :: mydevice_type

    mydevice_type = acc_get_device_type()

    if( mydevice_type == acc_device_nvidia) then
      print*, "openacc is running on gpus"
    elseif( mydevice_type == acc_device_host) then
      print*, "openacc is running on multiple cpus"
    else
      print*, "wrong acc device"
      stop
    endif

    n = 10000
    nloop = 1000
    
    n=n**2

    CALL CPU_TIME(t0)

    ALLOCATE (a(n), b(n), c(n))
    
    CALL acc_wait_all()
    CALL CPU_TIME(t1)
    a(:) = 0.0
    b(:) = 1.0
    c(:) = 1.0
    
    CALL CPU_TIME(t2)
    !$acc data copy(a, b, c)
    CALL acc_wait_all()
    CALL CPU_TIME(t3)
    CALL ArrayAdd2(a, b, c, n, nloop)
    CALL acc_wait_all()
    CALL CPU_TIME(t4)
    !$acc end data
    CALL acc_wait_all()
    CALL CPU_TIME(t5)

    PRINT*, " Plan A finished. "
    PRINT*, t4-t3, " secs by OpenACC"
    PRINT*, " Max(a) = ", MAXVAL(a)
    PRINT*, t0, t1, t2, t3, t4, t5

    PRINT*, " Add b and c to a ", nloop, " times. ", " They are arrays with length ", n

    ALLOCATE (a2(n), b2(n), c2(n))
    a2=0.0
    b2=1.0
    c2=1.0
    CALL CPU_TIME(t0)
    !$acc data copy(a2(:), b2(:), c2(:))
    CALL acc_wait_all()
    CALL CPU_TIME(t1)
    !$ACC kernels loop
    DO i = 1, n
      DO iloop = 1, nloop
        a2(i) = a2(i) + b2(i) + c2(i)
      END DO
    END DO
    CALL acc_wait_all()
    CALL CPU_TIME(t2)
    !$acc end data
    CALL acc_wait_all()
    CALL CPU_TIME(t3)

    PRINT*, " Plan B finished. "
    PRINT*, t2-t1, " secs by OpenACC"
    PRINT*, " Max(a) = ", MAXVAL(a2)
    PRINT*, t0, t1, t2, t3
    ! PRINT*, t_omp(1) - t_omp(1), t_omp(2) - t_omp(1), t_omp(3) - t_omp(1), t_omp(4) - t_omp(1)
    
    DEALLOCATE(a2, b2, c2)

  END SUBROUTINE openacc_multipurpose

  SUBROUTINE ArrayAdd2(a, b, c, n, nloop)
    IMPLICIT NONE
    INTEGER :: n, nloop
    INTEGER :: i, iloop
    REAL, ALLOCATABLE :: a(:), b(:), c(:)

    PRINT*, " Add b and c to a ", nloop, " times. ", " They are arrays with length ", n
    
    !$acc kernels loop
    DO i = 1, n
      DO iloop = 1, nloop
        a(i) = a(i) + b(i) + c(i)
        ! PRINT*, i, iloop, a(i), b(i)
      END DO
    END DO

  END SUBROUTINE ArrayAdd2

END MODULE
