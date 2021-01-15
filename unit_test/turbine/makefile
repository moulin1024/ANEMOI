################################################################################
# FC
################################################################################

FC = mpif90

################################################################################
# FFLAGS
################################################################################

FFLAGS = -O3 -Mpreprocess -mcmodel=medium
CUDA = -Mcuda -lcufft -ta=nvidia 
PRECISION=-DDOUBLE

################################################################################
# OBJ, PROG
################################################################################

OBJ = precision.o dimen.o control.o turbine_new_alm_check.o\
	  test.o \

PROG = wireles
all: build run
# --- compile ---
build:
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c ../../src/precision.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c dimen.cuf
	# $(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c turbine_gpu.cuf
	# $(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c turbine_lib_cpu.f90
	# $(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c turbine_cpu.f90
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c control.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c turbine_new_alm_check.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c test.cuf
	$(FC) $(FFLAGS) $(CUDA) $(OBJ) -o $(PROG).run $(LIBS)

run:
	mpirun -np 1 ./wireles.run
		
clean:
	find . -type f -name '*.o' -delete
	find . -type f -name '*.mod' -delete
	find . -type f -name '*.run' -delete
