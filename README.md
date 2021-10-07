# Sparse matrix benchmark for VirtualBow

This code tests some matrix operations with the [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) library that are relevant to [VirtualBow](https://github.com/bow-simulation/virtualbow). The results should help with the decision between sparse or dense matrices and how to use them most efficiently.

Background: VirtualBow currently uses dense matrices on the system level because they are easy to handle. However, most/all serious finite element solvers use sparse matrices for space and performance reasons. But the matrix dimensions that occur in VirtualBow are comparatively low (100-200) so it isn't immediately clear what's more efficient for this use case. Are sparse matrices worth the trouble? Read on to find out...

## Benchmarks

- **Sparse matrix update:** Test various strategies for updating the values in a sparse matrix with fixed sparsity pattern without reconstructing the matrix completely.

- **Matrix-vector multiplication:** Test the performance of matrix-vector multiplication for dense and sparse matrices. Test the influence of using multiple threads.

- **Matrix-matrix multiplication:** Test the performance of matrix-matrix multiplication for dense and sparse matrices. Test the influence of using multiple threads.

- **Linear system solving:** Test the performance of various dense and sparse matrix decompositions by solving a linear system.

## Conclusions

**Sparse vs dense matrices:** It seems that the dimension at which sparse matrix operations (multiplication and decomposition/solving) outperform the dense matrices lies at ~50 or below. At dimensions of 100-200, which are typical system dimensions in VirtualBow, the sparse matrices are already significantly faster. Multiple threads can speed up some of the dense operations, but not enough to offset this. So it seems that sparse matrices are the way to go, despite their practical disadvantages over dense matrices.

**Sparse matrix update:** The benchmark shows that storing the pointers to the elements of the sparse matrix is the fastest way of updating, while storing the indices is slightly slower but still much faster than any of the other methods.

**Linear system solving:** The best performance is shown by the ConjugatedGradient and BiCGSTAB methods. One might be slightly faster than the other, depending on the computer that the benchmark was run on. The [documentation for the ConjugateGradient method](https://eigen.tuxfamily.org/dox/classEigen_1_1ConjugateGradient.html) does not document the fact that the matrix must be positive definite, but [this stackoverflow post](https://stackoverflow.com/q/53010866) clarifies that this is indeed a requirement. Since the tangent stiffness matrices of mechanical systems are not necessarily positive definite, the BicGSTAB algorithm seems like the obvious choice here.
