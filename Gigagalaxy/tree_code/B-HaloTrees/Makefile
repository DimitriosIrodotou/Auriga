EXEC   = B-HaloTrees
OBJS   = main.o mymalloc.o read_parameters.o allvars.o io.o mergertree.o hdf5_read.o
INCL   = allvars.h  proto.h  Makefile

OPTIONS  += -DLONGIDS
OPTIONS  += -DSUBFIND_EXTRA      # read/include gas and GFM related group catalog fields
OPTIONS += -DIMPROVE_POINTERS
#OPTIONS += -DSKIP_SNAP
#OPTIONS  += -DSUBFIND_EXTRA_TNG  # read/include modified and new groupcat fields for TNG runs

SYSTYPE="magny"
#SYSTYPE="odyssey2"
#SYSTYPE="darwin"

HDF5INCL =  -DH5_USE_16_API
HDF5LIB  =  -lhdf5 -lz


ifeq ($(SYSTYPE),"magny")
CC       = mpicc
OPTIMIZE = -g -Wall
HDF5INCL = -I/hits/tap/sw/libs/hdf5-1.8.10/include -DH5_USE_16_API
HDF5LIB = -L/hits/tap/sw/libs/hdf5-1.8.10/lib -lhdf5 -lz
endif

ifeq ($(SYSTYPE),"darwin")
CC       = mpicc
OPTIMIZE = -g -Wall
HDF5INCL = -I/opt/local/inlcude -DH5_USE_16_API
HDF5LIB  = -L/opt/local/lib -lhdf5 -lz
endif

ifeq ($(SYSTYPE),"odyssey2")
CC       = mpicc
OPTIMIZE = -std=c11 -O3 -g -Wall -Wno-unused-but-set-variable -Wno-uninitialized -Wno-unknown-pragmas -Wno-unused-function -mprefer-avx128 -march=native
endif


LIBS   =   -lm $(HDF5LIB) -g
CFLAGS =   -g $(OPTIONS)  $(OPTIMIZE)  $(HDF5INCL)

$(EXEC): $(OBJS) 
	$(CC) $(OPTIMIZE) $(OBJS)  $(LIBS)   -o  $(EXEC)  

$(OBJS): $(INCL) 

clean:
	rm -f $(OBJS) $(EXEC)







