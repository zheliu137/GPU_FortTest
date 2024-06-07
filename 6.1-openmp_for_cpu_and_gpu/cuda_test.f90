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

  SUBROUTINE openmp_multipurpose
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
      print*, "Program is running on gpus"
    elseif( mydevice_type == acc_device_host) then
      print*, "Program is running on multiple cpus"
    else
      print*, "wrong acc device"
      stop
    endif

    print*, "number of threads?", omp_get_max_threads()

    n = 1000
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
!$omp target data map(tofrom:a,b,c)
!$omp barrier
    CALL CPU_TIME(t3)
    CALL ArrayAdd2(a, b, c, n, nloop)
!$omp barrier
    CALL CPU_TIME(t4)
!$omp end target data
!$omp barrier
    CALL CPU_TIME(t5)

    PRINT*, " Plan A finished. "
    PRINT*, t4-t3, " secs by OpenMP"
    PRINT*, " Max(a) = ", MAXVAL(a)
    PRINT*, t0, t1, t2, t3, t4, t5

    PRINT*, " Add b and c to a ", nloop, " times. ", " They are arrays with length ", n

    ALLOCATE (a2(n*10), b2(n*10), c2(n*10))
    a2=0.0
    b2=1.0
    c2=1.0
    CALL CPU_TIME(t0)
!$omp target data map(tofrom:a2(:),b2(:),c2(:))
!$omp barrier    
    CALL CPU_TIME(t1)
!$omp barrier    
!$omp target teams loop
    DO i = 1, n
      DO iloop = 1, nloop
        a2(i) = a2(i) + b2(i) + c2(i)
      END DO
    END DO
!$omp barrier    
    CALL CPU_TIME(t2)
!$omp end target data
!$omp barrier    
    CALL CPU_TIME(t3)

    PRINT*, " Plan B finished. "
    PRINT*, t2-t1, " secs by OpenMP"
    PRINT*, " Max(a) = ", MAXVAL(a2)
    PRINT*, t0, t1, t2, t3
    PRINT*, t1 - t0, t2 - t0, t3 - t0
    
    DEALLOCATE(a2, b2, c2)

  END SUBROUTINE openmp_multipurpose

  SUBROUTINE ArrayAdd2(a, b, c, n, nloop)
    IMPLICIT NONE
    INTEGER :: n, nloop
    INTEGER :: i, iloop
    REAL, ALLOCATABLE :: a(:), b(:), c(:)

    PRINT*, " Add b and c to a ", nloop, " times. ", " They are arrays with length ", n
    
!$omp target teams loop
    DO i = 1, n
      DO iloop = 1, nloop
        a(i) = a(i) + b(i) + c(i)
        ! PRINT*, i, iloop, a(i), b(i)
      END DO
    END DO

  END SUBROUTINE ArrayAdd2

END MODULE

! Code was translated using: /home/liuz/workplace/intel-application-migration-tool-for-openacc-to-openmp/src/intel-application-migration-tool-for-openacc-to-openmp cuda_test.f90
