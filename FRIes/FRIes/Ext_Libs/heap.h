/*! \file
 * \brief Utilities for constructng and maintaining a heap using an
 * auxiliary array of indices.
 *
 * Source code adapted from https://en.wikipedia.org/wiki/Heapsort
 */


#ifndef heap_h
#define heap_h

#define iLeftChild(i) (2 * i + 1)
#define iRightChild(i) (2 * i + 2)
#define iParent(i) ((i-1) / 2)

#include <stdio.h>
#include <math.h>

void sift_down(double *a, size_t *indices, size_t start, size_t end);
void heapify(double *a, size_t *indices, size_t count);

#endif /* heap_h */
