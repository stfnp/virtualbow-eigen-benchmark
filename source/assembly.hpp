#pragma once
#include "types.hpp"

// Element with a local, dense matrix and a list of indices that
// relate the local matrix entries to that of the system matrix
struct Element {
    DenseMatrix matrix;
    IndexVector indices;

    // Dimension of the element matrix and start index in the global system
    Element(int dimension, int start_index);

    // Generates a random, symmetric, positive definite matrix with dimension n
    // Source: https://math.stackexchange.com/a/358092
    static DenseMatrix create_matrix(int n);

    // Generates an ascending list of indices with size 'dimension', starting with start_index
    static IndexVector create_indices(int dimension, int start_index);

    // Creates n_elements elements of dimension dimension.
    // The element matrices are intersecting block diagonal matrices in the global matrix
    static std::vector<Element> create_elements(int dimension, int n_elements);
};

// System classes that take a list of elements and assemble a global matrix from the element matrices and indices
// using various different strategies

// Naive method
// Just calling coeffRef() like there is no tomorrow
struct System_CoeffRef {
    SparseMatrix matrix;
    std::vector<Element> elements;

    System_CoeffRef(const std::vector<Element>& elements);
    const SparseMatrix& assemble();
};

// Triplet method
// Collecting elements into a list of triplets and using setFromTriplets
struct System_Triplets {
    SparseMatrix matrix;
    std::vector<Element> elements;
    std::vector<Triplet> triplets;

    System_Triplets(const std::vector<Element>& elements);
    const SparseMatrix& assemble();
};

// Transformation method
// Use matrix operations to transform element matrices to global coordinates and then just sum them up
// Idea: Look at PermutationMatrix, it might be faster
struct System_Transform {
    SparseMatrix matrix;
    std::vector<Element> elements;
    std::vector<SparseMatrix> transforms;
    std::vector<SparseMatrix> transforms_t;

    System_Transform(const std::vector<Element>& elements);
    const SparseMatrix& assemble();
};

// Pointer method
// For each element, store the pointers of the global matrix entries associated with the local element entries
struct System_Pointers {
    SparseMatrix matrix;
    std::vector<Element> elements;
    std::vector<std::vector<Real*>> pointers;

    System_Pointers(const std::vector<Element>& elements);
    const SparseMatrix& assemble();
};

// Indices method
// Somewhat like the pointer method, but storing indices instead
struct System_Indices {
    SparseMatrix matrix;
    std::vector<Element> elements;
    std::vector<IndexVector> indices;

    System_Indices(const std::vector<Element>& elements);
    const SparseMatrix& assemble();

    // Copied and modified from Eigens SparseMatrix::coeff(Index row, Index col)
    // Assumption: Matrix is compressed and row major
    static long coeffIndex(const SparseMatrix& matrix, int row, int col);
};
