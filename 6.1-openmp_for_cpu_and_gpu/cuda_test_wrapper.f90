module cuda_wrappers
    contains
    subroutine cuda_wrapper()
        USE openacc
        USE cudamod, ONLY : openmp_multipurpose
!        CALL testcudalib()
!        CALL testcudaker()
        CALL openmp_multipurpose()

    end subroutine
end module