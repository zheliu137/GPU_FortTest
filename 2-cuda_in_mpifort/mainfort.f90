program main_program
    use cuda_wrappers, ONLY: cuda_wrapper
    implicit none
    include 'mpif.h'
    integer :: myid, ierr, nproc
    call MPI_INIT( ierr )
    
    call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
    
    call MPI_COMM_SIZE( MPI_COMM_WORLD, nproc, ierr )
    
    ! if( myid == 0 )CALL cuda_wrapper()
    CALL cuda_wrapper(myid)
    write(*,'("Hello World! Process ", I2, " in ", I5, " processors")') myid, nproc
    
    call MPI_FINALIZE(ierr)
        
end program