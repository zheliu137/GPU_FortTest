program main_program
    use cuda_wrappers, ONLY: cuda_wrapper
    USE OMP_LIB
    implicit none
    include 'mpif.h'
    integer :: myid, ierr, nproc
    INTEGER :: thread_id
    call MPI_INIT( ierr )
    
    call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
    
    call MPI_COMM_SIZE( MPI_COMM_WORLD, nproc, ierr )
    
    ! if( myid == 0 )CALL cuda_wrapper()
    CALL cuda_wrapper(myid)

    write(*,'("Hello World! Process ", I2, " in ", I5, " processors")') myid, nproc
    
    PRINT*, "Hello from process: ", myid, OMP_GET_THREAD_NUM()
    PRINT*, "Max OMP threads: ", myid, OMP_GET_MAX_THREADS()

    CALL omp_set_num_threads(10)
    write (*,*) ' Available Processors: ', omp_get_num_procs()
    write (*,*) ' Available Threads:    ', omp_get_max_threads()
    write (*,*) ' Threads in use:       ', omp_get_num_threads()
    PRINT *, "Max number of threads: ", OMP_GET_MAX_THREADS()
!$OMP PARALLEL 

    PRINT *, "Hello from process: ", OMP_GET_THREAD_NUM()

!$OMP END PARALLEL

    call MPI_FINALIZE(ierr)

end program