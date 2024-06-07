module cuda_wrappers
    contains
    subroutine cuda_wrapper()
        USE openacc
        USE cudamod, ONLY : testcudalib, testcudaker, testopenacc, openacc_multipurpose
!        CALL testcudalib()
!        CALL testcudaker()
        CALL openacc_multipurpose()

    end subroutine
end module