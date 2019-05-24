// Shamelessly copied from https://en.wikipedia.org/wiki/Heapsort
// but added indices so sort does not need to be performed in-place
// and modified so that elements are sorted by magnitude
// and added subroutines that treat hierarchical vectors

#ifndef heap_h
#define heap_h
#define iLeftChild(i) (2 * i + 1)
#define iRightChild(i) (2 * i + 2)
#define iParent(i) ((i-1) / 2)

#include <stdio.h>
#include <math.h>

void sift_down(double *a, size_t *indices, size_t start, size_t end);
void heapify(double *a, size_t *indices, size_t count);

//void sift_down_subd(double *vec, unsigned int *counts, double )

#endif /* heap_h */
