#pragma once
#include "assembly.hpp"

SparseMatrix<double> create_system_matrix(int n_elements, int n_dimension) {
    std::vector<Element> elements = create_elements(n_dimension, n_elements);
    System_Triplets system(elements);
    return system.assemble();
}

// Generates a random vector with dimension n
VectorXd create_random_vector(int n) {
    return VectorXd::Random(n);
}

template<typename MatrixType>
double matrix_vector_multiplication(int n_elements, int n_dimension, int samples) {
    MatrixType matrix = create_system_matrix(n_elements, n_dimension);
    VectorXd vector = create_random_vector(matrix.rows());
    VectorXd result;

    return execute_and_measure([&] {
        result = matrix*vector;
    }, samples);
}

template<typename MatrixType>
double matrix_matrix_multiplication(int n_elements, int n_dimension, int samples) {
    MatrixType matrix1 = create_system_matrix(n_elements, n_dimension);
    MatrixType matrix2 = create_system_matrix(n_elements, n_dimension);
    MatrixType result;

    return execute_and_measure([&] {
        result = matrix1*matrix2;
    }, samples);
}

template<typename MatrixType, typename DecompType>
double solve_linear_system(int n_elements, int n_dimension, int samples) {
    MatrixType matrix = create_system_matrix(n_elements, n_dimension);
    VectorXd vector = create_random_vector(matrix.rows());
    VectorXd result;

    DecompType decomp;

    return execute_and_measure([&] {
        decomp.compute(matrix);
        result = decomp.solve(vector);
    }, samples);
}
