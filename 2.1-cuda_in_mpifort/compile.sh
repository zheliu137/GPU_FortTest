pgf90 -c cmod1.cuf 
ifort -c test.f90 
pgf90 -c main.f90 
pgf90 -Mcuda main.o test.o cmod1.o -L/opt/intel/oneapi/compiler/latest/lib/ -lifport -lifcore -limf
./a.out