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

  SUBROUTINE openmp_lib_offl
    !
    IMPLICIT NONE
    !
    INTEGER :: n, nloop
    REAL, ALLOCATABLE :: a(:), b(:), c(:)
    REAL, ALLOCATABLE :: a2(:), b2(:), c2(:)
    REAL :: t0, t1, t2, t3, t4, t5
    REAL :: t_omp(10)
    INTEGER :: i, j, k, iloop
    INTEGER :: sp
    INTEGER :: max_idx(2)
    !
    REAL(kind=DP)        :: h_a(200)
    REAL(kind=DP)        :: h_b(200)
    INTEGER :: istat
    REAL :: time
    ! integer(kind=acc_device_kind) :: mydevice_type

    ! mydevice_type = acc_get_device_type()

    ! if( mydevice_type == acc_device_nvidia) then
    !   print*, "openacc is running on gpus"
    ! elseif( mydevice_type == acc_device_host) then
    !   print*, "openacc is running on multiple cpus"
    ! else
    !   print*, "wrong acc device"
    !   stop
    ! endif
    ! print*, default-device-var, target-offload-var

    n = 1000
    nloop = 1
    
    n=n**2

    CALL CPU_TIME(t0)

    ALLOCATE (a(n), b(n), c(n))
    
    CALL CPU_TIME(t1)
    a(:) = 0.0
    b(:) = 1.0
    c(:) = 1.5
    
    CALL CPU_TIME(t2)
    !$omp target data map(tofrom: a(:), b(:), c(:))
    !$omp do
    do i = 1, n/100
      ! CALL arrayadd_inline(a(i), b(i), c(i), nloop)
      CALL sgemv('N', 10, 10, 1.0, c((i-1)*100+1), 10, b((i-1)*10+1), 10, 0.0, a((i-1)*10+1), 10)
    enddo
    !$omp end do
    !$omp end target data 
    CALL CPU_TIME(t3)

    PRINT*, " max(a) = ", MAXVAL(a)
    PRINT*, " min(a) = ", MINVAL(a)
    PRINT*, " time cost = ", t3 - t2

    DEALLOCATE (a, b, c)

  END SUBROUTINE openmp_lib_offl

  SUBROUTINE arrayadd_inline(a, b, c, nloop)
    IMPLICIT NONE
    INTEGER :: nloop
    INTEGER :: i, iloop
    REAL :: a, b, c

    DO iloop = 1, nloop
      a = a + b + c
    END DO

  END SUBROUTINE arrayadd_inline

END MODULE
