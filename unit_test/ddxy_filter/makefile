################################################################################
# FC
################################################################################

FC = mpif90

################################################################################
# FFLAGS
################################################################################

FFLAGS = -O3 -Mpreprocess
CUDA = -Mcuda -lcufft -ta=nvidia 
PRECISION=-DDOUBLE

################################################################################
# OBJ, PROG
################################################################################

OBJ = precision.o dimen.o cufft_m.o filter.o update.o\
	  test.o \

PROG = wireles
all: build run
# --- compile ---
build:
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c ../../src/precision.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c ../common/dimen.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c ../../src/cufft_m.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c ../../src/filter.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c ../../src/update.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c test.cuf
	$(FC) $(FFLAGS) $(CUDA) $(OBJ) -o $(PROG).run $(LIBS)

run:
	mpirun -np 1 ./wireles.run
		

clean:
	find . -type f -name '*.o' -delete
	find . -type f -name '*.mod' -delete
	find . -type f -name '*.run' -delete
