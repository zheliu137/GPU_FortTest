subroutine test
    implicit none
    real  :: A(10,10), B(10,10), C(10,10)
    A = 10.1
    B = 0.11
    C = 0.
    call cmod1_mmul(A,B,C,10,10)
    print *, A(1,2), B(2,4), C(3,6)
   
end subroutine test 