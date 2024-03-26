INTEL_PATH=/opt/intel/oneapi/

NVFORT = nvfortran
F90 = ifort

NVFFLAGS = -cuda

test.x: 
	$(NVFORT) -c cmod1.cuf 
	$(F90) -c test.f90 
	$(NVFORT) -c main.f90 
	$(NVFORT) $(NVFFLAGS) main.o test.o cmod1.o -L$(INTEL_PATH)/compiler/latest/lib/ -lifport -lifcore -limf -o $@
	./$@

clean:
	rm -f *.o *.mod test.x