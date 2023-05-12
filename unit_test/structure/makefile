################################################################################
# FC
################################################################################

FC = mpif90

################################################################################
# FFLAGS
################################################################################

FFLAGS = -O3 -Mpreprocess
CUDA = -Mcuda -lcufft -ta=nvidia 

################################################################################
# OBJ, PROG
################################################################################

OBJ = precision.o dimen.o structural.o\
	  test.o \

PROG = wireles
all: build run
# --- compile ---
build:
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c ../../src/precision.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c ../common/dimen.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c ../../src/structural.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c test.cuf
	$(FC) $(FFLAGS) $(CUDA) $(OBJ) -o $(PROG).run $(LIBS)

run:
	mpirun -np 2 ./wireles.run
		

clean:
	find . -type f -name '*.o' -delete
	find . -type f -name '*.mod' -delete
	find . -type f -name '*.run' -delete
