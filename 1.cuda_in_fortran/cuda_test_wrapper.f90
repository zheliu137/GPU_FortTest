module cuda_wrappers
    contains
    subroutine cuda_wrapper()
        use cudamod, only : testcudalib, testcudaker
        CALL testcudalib()
        CALL testcudaker()
    end subroutine
end module