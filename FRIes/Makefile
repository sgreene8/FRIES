CFLAGS = -Wall -O3 -ffast-math
CC = clang

UTILS = \
    compress_utils.o\
    csvparser.o\
    det_store.o\
    fci_utils.o\
    io_utils.o\
    near_uniform.o\
    heap.o\
    argparse.o\
    vec_utils.o\
    math_utils.o\
    hub_holstein.o\
    heat_bathPP.o

fciqmc: $(UTILS) main.c mpi_switch.h
	$(CC) $(CFLAGS) -o fciqmc.x $(UTILS) main.c -L. -ldcmt -lmpi

compress_utils.o : compress_utils.c compress_utils.h mpi_switch.h
	$(CC) -c $(CFLAGS) compress_utils.c

csvparser.o : csvparser.c csvparser.h
	$(CC) -c $(CFLAGS) csvparser.c

det_store.o : det_store.c det_store.h
	$(CC) -c $(CFLAGS) det_store.c

fci_utils.o : fci_utils.c fci_utils.h
	$(CC) -c $(CFLAGS) fci_utils.c

io_utils.o : io_utils.c io_utils.h mpi_switch.h
	$(CC) -c $(CFLAGS) io_utils.c

near_uniform.o : near_uniform.c near_uniform.h
	$(CC) -c $(CFLAGS) near_uniform.c

heap.o : heap.h heap.c
	$(CC) -c $(CFLAGS) heap.c

argparse.o : argparse.h argparse.c
	$(CC) -c $(CFLAGS) argparse.c

vec_utils.o : vec_utils.h vec_utils.c mpi_switch.h
	$(CC) -c $(CFLAGS) vec_utils.c

math_utils.o : math_utils.h math_utils.c
	$(CC) -c $(CFLAGS) math_utils.c

hub_holstein.o : hub_holstein.h hub_holstein.c
	$(CC) -c $(CFLAGS) hub_holstein.c

heat_bathPP.o : heat_bathPP.h heat_bathPP.c
	$(CC) -c $(CFLAGS) heat_bathPP.c

clean:
	rm *.o fciqmc.x
