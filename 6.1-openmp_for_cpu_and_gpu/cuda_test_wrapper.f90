module cuda_wrappers
    contains
    subroutine cuda_wrapper()
        USE openacc
        USE cudamod, ONLY : openmp_multicpus, openmp_gpu_offload
!        CALL testcudalib()
!        CALL testcudaker()
        CALL openmp_multicpus()
        CALL openmp_gpu_offload()

    end subroutine
end module