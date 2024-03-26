program main_program
    use cuda_wrappers, only: cuda_wrapper

    CALL cuda_wrapper()
    ! use cudamod, only : testcudalib, testcudaker
    ! integer :: mpime
    ! CALL testcudalib()
    ! CALL testcudaker()

end program