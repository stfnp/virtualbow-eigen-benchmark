#include "assembly.hpp"

// Dimension of the element matrix and start index in the global system
Element::Element(int dimension, int start_index)
    : matrix(create_matrix(dimension)),
      indices(create_indices(dimension, start_index))
{
    if(dimension % 2 != 0) {
        throw std::invalid_argument("Element dimension must be an even number");
    }
}

// Generates a random, symmetric, positive definite matrix with dimension n
// Source: https://math.stackexchange.com/a/358092
DenseMatrix Element::create_matrix(int n) {
    DenseMatrix A = 0.5*(DenseMatrix::Random(n, n) + DenseMatrix::Identity(n, n));    // Matrix with elements in [0, 1]
    return 0.5*(A.transpose() + A) + n*DenseMatrix::Identity(n, n);    // Construct symmetry and diagonal dominance
}

// Generates an ascending list of indices with size 'dimension', starting with start_index
IndexVector Element::create_indices(int dimension, int start_index) {
    return IndexVector::LinSpaced(dimension, start_index, start_index + dimension - 1);
}

std::vector<Element> Element::create_elements(int dimension, int n_elements) {
    std::vector<Element> elements;
    for(int i = 0; i < n_elements; ++i) {
        int start_index = i*dimension/2;
        elements.push_back(Element(dimension, start_index));
    }
    return elements;
}

System_CoeffRef::System_CoeffRef(const std::vector<Element>& elements)
    : elements(elements)
{
    //Determine matrix size
    int max_index = 0;
    for(auto& e: elements) {
        max_index = std::max(max_index, e.indices.maxCoeff());
    }

    // Initialize matrix
    this->matrix = SparseMatrix(max_index + 1, max_index + 1);
}

const SparseMatrix& System_CoeffRef::assemble() {
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

System_Triplets::System_Triplets(const std::vector<Element>& elements)
    : elements(elements)
{
    //Determine matrix size
    int max_index = 0;
    for(auto& e: elements) {
        max_index = std::max(max_index, e.indices.maxCoeff());
    }

    // Initialize matrix
    matrix = SparseMatrix(max_index + 1, max_index + 1);
}

const SparseMatrix& System_Triplets::assemble() {
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

System_Transform::System_Transform(const std::vector<Element>& elements)
    : elements(elements)
{
    //Determine matrix size
    int max_index = 0;
    for(auto& e: elements) {
        max_index = std::max(max_index, e.indices.maxCoeff());
    }

    // Initialize matrix
    this->matrix = SparseMatrix(max_index + 1, max_index + 1);

    // Initialize transforms
    for(auto& e: elements) {
        SparseMatrix transform(max_index + 1, e.indices.size());    // Global dimension x element dimension
        SparseMatrix transform_t(e.indices.size(), max_index + 1);    // Element dimension x global dimension

        for(int i = 0; i < e.indices.size(); ++i) {
            int j = e.indices(i);
            transform.insert(j, i) = 1.0;
            transform_t.insert(i, j) = 1.0;
        }
        transforms.push_back(transform);
        transforms_t.push_back(transform_t);
    }
}

const SparseMatrix& System_Transform::assemble() {
    matrix.coeffs().setZero();
    SparseMatrix temp;
    for(int i = 0; i < elements.size(); ++i) {
        temp = elements[i].matrix.sparseView();    // TODO: Write without this
        matrix += transforms[i] * temp * transforms_t[i];
    }
    return matrix;
}

System_Pointers::System_Pointers(const std::vector<Element>& elements)
    : elements(elements)
{
    //Determine matrix size
    int max_index = 0;
    for(auto& e: elements) {
        max_index = std::max(max_index, e.indices.maxCoeff());
    }

    // Initialize sparsity pattern with zeros
    std::vector<Triplet> triplets;
    for(auto& e: elements) {
        for(int i = 0; i < e.matrix.rows(); ++i) {
            for(int j = 0; j < e.matrix.cols(); ++j) {
                triplets.emplace_back(e.indices(i), e.indices(j), 0.0);
            }
        }
    }
    matrix = SparseMatrix(max_index + 1, max_index + 1);
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    // Store coefficient references
    for(auto& e: elements) {
        std::vector<Real*> p;
        for(int i = 0; i < e.matrix.rows(); ++i) {
            for(int j = 0; j < e.matrix.cols(); ++j) {
                p.push_back(&matrix.coeffRef(e.indices(i), e.indices(j)));
            }
        }
        pointers.push_back(p);
    }
}

const SparseMatrix& System_Pointers::assemble() {
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

System_Indices::System_Indices(const std::vector<Element>& elements)
    : elements(elements)
{
    //Determine matrix size
    int max_index = 0;
    for(auto& e: elements) {
        max_index = std::max(max_index, e.indices.maxCoeff());
    }

    // Initialize sparsity pattern with zeros
    std::vector<Triplet> triplets;
    for(auto& e: elements) {
        for(int i = 0; i < e.matrix.rows(); ++i) {
            for(int j = 0; j < e.matrix.cols(); ++j) {
                triplets.emplace_back(e.indices(i), e.indices(j), 0.0);
            }
        }
    }
    matrix = SparseMatrix(max_index + 1, max_index + 1);
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    // Initialize coefficient indices
    for(auto& e: elements) {
        IndexVector element_indices(e.matrix.size());
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

const SparseMatrix& System_Indices::assemble() {
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

long System_Indices::coeffIndex(const SparseMatrix& matrix, int row, int col) {
    int start = matrix.outerIndexPtr()[row];
    int end = matrix.outerIndexPtr()[row + 1];
    long i = matrix.data().searchLowerIndex(start, end - 1, col);

    if((i < end) && (matrix.data().index(i) == col)) {
        return i;
    } else {
        throw std::invalid_argument("Value does not exist");
    }
}

