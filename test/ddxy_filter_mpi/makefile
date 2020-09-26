################################################################################
# FC
################################################################################

FC = pgf90

################################################################################
# FFLAGS
################################################################################

FFLAGS = -O3 -Mpreprocess -mcmodel=medium
CUDA = -Mcuda -lcufft -ta=nvidia
PRECISION=-DDOUBLE

################################################################################
# OBJ, PROG
################################################################################

OBJ = cpu_timer.o precision.o dimen.o cufft_m.o filter_stacked.o \
	  test.o \

PROG = wireles
all: build run
# --- compile ---
build:
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c ../../src/precision.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c ../common/cpu_timer.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c ../common/dimen.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c ../../src/cufft_m.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c filter_stacked.cuf
	# $(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c ../cpu_benchmark/ddxy_filter_cpu.cuf
	$(FC) $(FFLAGS) $(CUDA) $(PRECISION) -c test.cuf
	$(FC) $(FFLAGS) $(CUDA) $(OBJ) -o $(PROG).run

run:
	./wireles.run
		

clean:
	find . -type f -name '*.o' -delete
	find . -type f -name '*.mod' -delete
	find . -type f -name '*.run' -delete