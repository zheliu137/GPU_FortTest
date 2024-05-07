module cuda_wrappers
    contains
    subroutine cuda_wrapper()
        USE openacc
        USE cudamod, ONLY : testcudalib, testcudaker, testopenacc
        CALL testcudalib()
        CALL testcudaker()
        CALL testopenacc()

    end subroutine
end module