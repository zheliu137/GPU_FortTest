PROGRAM main
    ! use cuda_wrappers, ONLY: cuda_wrapper
    USE OMP_LIB
    include 'mpif.h'
    call MPI_INIT( ierr )
    
    call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
    
    call MPI_COMM_SIZE( MPI_COMM_WORLD, nproc, ierr )
    
!     if( myid == 0 )CALL cuda_wrapper(myid)
    CALL cuda_wrapper(myid)

    CALL omp_set_num_threads(10)
    write (*,*) ' Available Processors: ', omp_get_num_procs()
    write (*,*) ' Available Threads:    ', omp_get_max_threads()
    write (*,*) ' Threads in use:       ', omp_get_num_threads()
    PRINT *, "Max number of threads: ", OMP_GET_MAX_THREADS()
!$OMP PARALLEL 
    
    PRINT *, "Hello from process: ", OMP_GET_THREAD_NUM(), myid
    
!$OMP END PARALLEL
    
    call MPI_FINALIZE(ierr)

END