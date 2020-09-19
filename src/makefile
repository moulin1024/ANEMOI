################################################################################
# FC
################################################################################

FC = pgf90

################################################################################
# FFLAGS
################################################################################

FFLAGS = -O3 -Mpreprocess -mcmodel=medium
CUDA = -Mcuda -lcufft -ta=nvidia
# debug
# -check bounds -debug all -traceback -fpe0 fp-stack-check -debug inline-debug-info -parallel-source-info=2 -warn all

# prof
# -pg

################################################################################
# LIBS
################################################################################

# LIBS = -L/home/moulin/anaconda3/lib -lfftw3
# INC = #-I/opt/intel/compilers_and_libraries_2019.4.243/linux/mpi/intel64/include/
################################################################################
# OBJ, PROG
################################################################################

OBJ = precision.o dimen.o cufft_m.o io.o filter.o derivative.o \
	  wall.o convection.o sgs.o rhs.o turbine.o  pressure.o divergence.o \
	  time.o main.o \

PROG = wireles_src

# --- compile ---
 %.o:%.cuf
		$(FC) $(FFLAGS) $(CUDA) $(INC) $(PRECISION) -c $< -o $@
# --- link ---
$(PROG):$(OBJ)
	$(FC) $(FFLAGS) $(CUDA) $(OBJ) -o $(PROG) $(INC) $(LIBS)

# --- clean ---
clean:
	rm -f *.o *.bck *.bak *.mod $(PROG)
