#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

using Eigen::SparseMatrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::Triplet;

// Element with a local, dense matrix and a list of indices that
// relate the local matrix entries to that of the system matrix
struct Element {
    MatrixXd matrix;
    VectorXi indices;

    // Dimension of the element matrix and start index in the global system
    Element(int dimension, int start_index)
        : matrix(create_matrix(dimension)),
          indices(create_indices(dimension, start_index))
    {
        if(dimension % 2 != 0) {
            throw std::invalid_argument("Element dimension must be an even number");
        }
    }

    // Generates a random, symmetric, positive definite matrix with dimension n
    // Source: https://math.stackexchange.com/a/358092
    static MatrixXd create_matrix(int n) {
        MatrixXd A = 0.5*(MatrixXd::Random(n, n) + MatrixXd::Identity(n, n));    // Matrix with elements in [0, 1]
        return 0.5*(A.transpose() + A) + n*MatrixXd::Identity(n, n);    // Construct symmetry and diagonal dominance
    }

    // Generates an ascending list of indices with size 'dimension', starting with start_index
    static VectorXi create_indices(int dimension, int start_index) {
        return VectorXi::LinSpaced(dimension, start_index, start_index + dimension - 1);
    }
};

// System classes that take a list of elements and assemble a global matrix from the element matrices and indices
// using various different strategies

// Naive method
// Just calling coeffRef() like there is no tomorrow
struct System_CoeffRef {
    SparseMatrix<double> matrix;
    std::vector<Element> elements;

    System_CoeffRef(const std::vector<Element>& elements)
        : elements(elements)
    {
        //Determine matrix size
        int max_index = 0;
        for(auto& e: elements) {
            max_index = std::max(max_index, e.indices.maxCoeff());
        }

        // Initialize matrix
        this->matrix = SparseMatrix<double>(max_index + 1, max_index + 1);
    }

    const SparseMatrix<double>& assemble() {
        matrix.coeffs().setZero();    // Faster than matrix.setZero(), but matrix needs to be compressed (see makeCompressed() below)
        for(auto& e: elements) {
            for(int i = 0; i < e.matrix.rows(); ++i) {
                for(int j = 0; j < e.matrix.cols(); ++j) {
                    matrix.coeffRef(e.indices(i), e.indices(j)) += e.matrix(i, j);
                }
            }
        }
        matrix.makeCompressed();
        return matrix;
    }
};

// Triplet method
// Collecting elements into a list of triplets and using setFromTriplets
struct System_Triplets {
    SparseMatrix<double> matrix;
    std::vector<Element> elements;
    std::vector<Triplet<double>> triplets;

    System_Triplets(const std::vector<Element>& elements)
        : elements(elements)
    {
        //Determine matrix size
        int max_index = 0;
        for(auto& e: elements) {
            max_index = std::max(max_index, e.indices.maxCoeff());
        }

        // Initialize matrix
        matrix = SparseMatrix<double>(max_index + 1, max_index + 1);
    }

    const SparseMatrix<double>& assemble() {
        triplets.clear();
        for(auto& e: elements) {
            for(int i = 0; i < e.matrix.rows(); ++i) {
                for(int j = 0; j < e.matrix.cols(); ++j) {
                    triplets.emplace_back(e.indices(i), e.indices(j), e.matrix(i, j));
                }
            }
        }
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        return matrix;
    }
};

// Transformation method
// Use matrix operations to transform element matrices to global coordinates and then just sum them up
// Idea: Look at PermutationMatrix, it might be faster
struct System_Transform {
    SparseMatrix<double> matrix;
    std::vector<Element> elements;
    std::vector<SparseMatrix<double>> transforms;
    std::vector<SparseMatrix<double>> transforms_t;

    System_Transform(const std::vector<Element>& elements)
        : elements(elements)
    {
        //Determine matrix size
        int max_index = 0;
        for(auto& e: elements) {
            max_index = std::max(max_index, e.indices.maxCoeff());
        }

        // Initialize matrix
        this->matrix = SparseMatrix<double>(max_index + 1, max_index + 1);

        // Initialize transforms
        for(auto& e: elements) {
            SparseMatrix<double> transform(max_index + 1, e.indices.size());    // Global dimension x element dimension
            SparseMatrix<double> transform_t(e.indices.size(), max_index + 1);    // Element dimension x global dimension

            for(int i = 0; i < e.indices.size(); ++i) {
                int j = e.indices(i);
                transform.insert(j, i) = 1.0;
                transform_t.insert(i, j) = 1.0;
            }
            transforms.push_back(transform);
            transforms_t.push_back(transform_t);
        }
    }

    const SparseMatrix<double>& assemble() {
        matrix.coeffs().setZero();
        SparseMatrix<double> temp;
        for(int i = 0; i < elements.size(); ++i) {
            temp = elements[i].matrix.sparseView();    // TODO: Write without this
            matrix += transforms[i] * temp * transforms_t[i];
        }
        return matrix;
    }
};

// Pointer method
// For each element, store the pointers of the global matrix entries associated with the local element entries
struct System_Pointers {
    SparseMatrix<double> matrix;
    std::vector<Element> elements;
    std::vector<std::vector<double*>> pointers;

    System_Pointers(const std::vector<Element>& elements)
        : elements(elements)
    {
        //Determine matrix size
        int max_index = 0;
        for(auto& e: elements) {
            max_index = std::max(max_index, e.indices.maxCoeff());
        }

        // Initialize sparsity pattern with zeros
        std::vector<Triplet<double>> triplets;
        for(auto& e: elements) {
            for(int i = 0; i < e.matrix.rows(); ++i) {
                for(int j = 0; j < e.matrix.cols(); ++j) {
                    triplets.emplace_back(e.indices(i), e.indices(j), 0.0);
                }
            }
        }
        matrix = SparseMatrix<double>(max_index + 1, max_index + 1);
        matrix.setFromTriplets(triplets.begin(), triplets.end());

        // Store coefficient references
        for(auto& e: elements) {
            std::vector<double*> p;
            for(int i = 0; i < e.matrix.rows(); ++i) {
                for(int j = 0; j < e.matrix.cols(); ++j) {
                    p.push_back(&matrix.coeffRef(e.indices(i), e.indices(j)));
                }
            }
            pointers.push_back(p);
        }
    }

    const SparseMatrix<double>& assemble() {
        matrix.coeffs().setZero();
        for(size_t k = 0; k < elements.size(); ++k) {
            auto& e = elements[k];
            auto& r = pointers[k];

            // TODO: Use iterators here?
            for(int i = 0; i < e.matrix.size(); ++i) {
                *r[i] += e.matrix.reshaped()(i);
            }
        }
        return matrix;
    }
};

// Indices method
// Somewhat like the pointer method, but storing indices instead
struct System_Indices {
    SparseMatrix<double> matrix;
    std::vector<Element> elements;
    std::vector<VectorXi> indices;

    System_Indices(const std::vector<Element>& elements)
        : elements(elements)
    {
        //Determine matrix size
        int max_index = 0;
        for(auto& e: elements) {
            max_index = std::max(max_index, e.indices.maxCoeff());
        }

        // Initialize sparsity pattern with zeros
        std::vector<Triplet<double>> triplets;
        for(auto& e: elements) {
            for(int i = 0; i < e.matrix.rows(); ++i) {
                for(int j = 0; j < e.matrix.cols(); ++j) {
                    triplets.emplace_back(e.indices(i), e.indices(j), 0.0);
                }
            }
        }
        matrix = SparseMatrix<double>(max_index + 1, max_index + 1);
        matrix.setFromTriplets(triplets.begin(), triplets.end());

        // Initialize coefficient indices
        for(auto& e: elements) {
            VectorXi element_indices(e.matrix.size());
            int k = 0;
            for(int i = 0; i < e.matrix.rows(); ++i) {
                for(int j = 0; j < e.matrix.cols(); ++j) {
                    element_indices[k] = coeffIndex(matrix, e.indices(i), e.indices(j));
                    ++k;
                }
            }
            indices.push_back(element_indices);
        }
    }

    // Copied and modified from Eigens SparseMatrix::coeff(Index row, Index col)
    // Assumption: Matrix is compressed and row major
    static long coeffIndex(const SparseMatrix<double>& matrix, int row, int col) {
        int start = matrix.outerIndexPtr()[row];
        int end = matrix.outerIndexPtr()[row + 1];
        long i = matrix.data().searchLowerIndex(start, end - 1, col);

        if((i < end) && (matrix.data().index(i) == col)) {
            return i;
        } else {
            throw std::invalid_argument("Value does not exist");
        }
    }

    const SparseMatrix<double>& assemble() {
        matrix.coeffs().setZero();
        for(size_t i = 0; i < elements.size(); ++i) {
            matrix.coeffs()(indices[i]) += elements[i].matrix.array().reshaped();
            // TODO: Why is this slightly faster?
            // for(int k = 0; k < indices[i].size(); ++k) {
            //     matrix.coeffs()(k) += elements[i].matrix.reshaped()(k);
            // }
        }
        return matrix;
    }
};

// Creates n_elements elements of dimension dimension.
// The element matrices are intersecting block diagonal matrices in the global matrix
std::vector<Element> create_elements(int dimension, int n_elements) {
    std::vector<Element> elements;
    for(int i = 0; i < n_elements; ++i) {
        int start_index = i*dimension/2;
        elements.push_back(Element(dimension, start_index));
    }
    return elements;
}
