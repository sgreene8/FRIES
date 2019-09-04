#include "heap.h"


void sift_down(double *a, size_t *indices, size_t start, size_t end) {
    size_t root = start;
    size_t child, swap;
    double tmp;
    
    while (iLeftChild(root) <= end) {  // (While the root has at least one child)
        child = iLeftChild(root);  // (Left child of root)
        swap = root; // (Keeps track of child to swap with)
        
        if (fabs(a[indices[swap]]) < fabs(a[indices[child]])) {
            swap = child;
        }
        // (If there is a right child and that child is greater)
        if (child + 1 <= end && fabs(a[indices[swap]]) < fabs(a[indices[child+1]])) {
            swap = child + 1;
        }
        if (swap == root) {
            // (The root holds the largest element. Since we assume the heaps rooted at the
            // children are valid, this means that we are done.)
            return;
        }
        else {
            tmp = indices[root];
            indices[root] = indices[swap];
            indices[swap] = tmp;
            root = swap; // (repeat to continue sifting down the child now)
        }
    }
}

void heapify(double *a, size_t *indices, size_t count) {
    // (start is assigned the index in 'a' of the last parent node)
    // (the last element in a 0-based array is at index count-1; find the parent of that element)
    if (count == 1) {
        return;
    }
    ssize_t start = iParent(count - 1);
    
    while (start >= 0) {
        // (sift down the node at index 'start' to the proper place such that all nodes below
        // the start index are in heap order)
        sift_down(a, indices, start, count - 1);
        // (go to the next parent node)
        start = start - 1;
    }
    // (after sifting down the root all nodes/elements are in heap order)
}
