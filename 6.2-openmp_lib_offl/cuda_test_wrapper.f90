module cuda_wrappers
    contains
    subroutine cuda_wrapper()
        USE openacc
        USE cudamod, ONLY : openmp_lib_offl
!        CALL testcudalib()
!        CALL testcudaker()
        CALL openmp_lib_offl()

    end subroutine
end module