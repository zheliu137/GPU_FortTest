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

  SUBROUTINE openmp_gpu_offload
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

    print*, "Program is running on gpus"

    n = 1000
    nloop = 1000
    
    n=n**2

    CALL CPU_TIME(t0)

    ALLOCATE (a(n), b(n), c(n))
    
    CALL CPU_TIME(t1)
    a(1:n) = 0.0
    b(1:n) = 1.0
    c(1:n) = 2.0
    
    CALL CPU_TIME(t2)
!$omp target data map(tofrom: a, b, c)
!$omp barrier
    CALL CPU_TIME(t3)
!$omp target teams loop thread_limit(512) default(none) &
!$omp&         private( i, iloop ) &
!$omp&         shared( a, b, c, n, nloop )
    DO i = 1, n
      DO iloop = 1, nloop
        a(i) = a(i) + b(i) + c(i)
        ! PRINT*, i, iloop, a(i), b(i)
      END DO
    END DO
!$omp end target teams loop
!$omp barrier
    CALL CPU_TIME(t4)
!$omp end target data
!$omp barrier
    CALL CPU_TIME(t5)

    PRINT*, "Finished. "
    PRINT*, t4-t3, " secs by OpenMP"
    PRINT*, " Max(a) = ", MAXVAL(a)
    PRINT*, t0, t1, t2, t3, t4, t5
    
    DEALLOCATE(a, b, c)

  END SUBROUTINE openmp_gpu_offload

  SUBROUTINE openmp_multicpus
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


    print*, "Program is running on multiple cpus"

    n = 1000
    nloop = 1000
    
    n=n**2

    CALL CPU_TIME(t0)

    ALLOCATE (a(n), b(n), c(n))
    
    CALL CPU_TIME(t1)
    a(1:n) = 0.0
    b(1:n) = 1.0
    c(1:n) = 2.0
    
    CALL CPU_TIME(t2)
!$omp parallel do default(none) &
!$omp&         private( i, iloop ) &
!$omp&         shared( a, b, c, n, nloop )
    DO i = 1, n
      DO iloop = 1, nloop
        a(i) = a(i) + b(i) + c(i)
        ! PRINT*, i, iloop, a(i), b(i)
      END DO
    END DO
!$omp end parallel do

    CALL CPU_TIME(t3)

    PRINT*, "Finished. "
    PRINT*, t3-t2, " secs by OpenMP"
    PRINT*, " Max(a) = ", MAXVAL(a)
    PRINT*, t0, t1, t2, t3
    
    DEALLOCATE(a, b, c)

  END SUBROUTINE openmp_multicpus


END MODULE

! Code was translated using: /home/liuz/workplace/intel-application-migration-tool-for-openacc-to-openmp/src/intel-application-migration-tool-for-openacc-to-openmp cuda_test.f90
