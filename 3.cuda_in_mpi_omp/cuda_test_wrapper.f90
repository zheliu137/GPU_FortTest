module cuda_wrappers
    contains
    subroutine cuda_wrapper(mpime)
        use cudamod, only : testcudalib, testcudaker
        integer :: mpime
        CALL testcudalib(mpime)
        CALL testcudaker()
    end subroutine
end module
