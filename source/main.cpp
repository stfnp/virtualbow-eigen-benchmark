//#define EIGEN_DONT_PARALLELIZE

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <chrono>

using SparseMatrix = Eigen::SparseMatrix<double>;
using DenseMatrix = Eigen::MatrixXd;
using DenseVector = Eigen::VectorXd;

// Generates a random, symmetric, positive definite matrix with dimension n and bandwidth k
SparseMatrix create_matrix(int n, int k) {
    SparseMatrix matrix(n, n);

    for(int i = 0; i < n; ++i) {
        for(int j = i; j < n; ++j) {
            // Create random double value in [0, 1]
            double value = (double) rand()/RAND_MAX;    // https://stackoverflow.com/a/2704552

            // On diagonal elements, add term to make the matrix diagonally dominant
            // and positive definite (https://math.stackexchange.com/a/358092
            // On off-diagonal elements, add value for ij and ji for symmetry
            if(i == j) {
                matrix.coeffRef(i, j) = value + 2*k;
            }
            else if((j >= i - k) && (j <= i + k)) {
                matrix.coeffRef(i, j) = value;
                matrix.coeffRef(j, i) = value;
            }
        }
    }

    return matrix;
}

// Generates a random vector with dimension n
DenseVector create_vector(int n) {
    DenseVector vector(n);
    for(int i = 0; i < n; ++i) {
        vector(i) = (double) rand()/RAND_MAX;    // https://stackoverflow.com/a/2704552
    }

    return vector;
}

template<typename F>
double execute_and_measure(F function, int samples) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;

    auto t0 = high_resolution_clock::now();

    for(int i = 0; i < samples; ++i) {
        function();
    }

    auto t1 = high_resolution_clock::now();

    duration<double, std::milli> ms_double = t1 - t0;
    return ms_double.count();
}

template<typename MatrixType>
double matrix_vector_multiplication(int n, int k, int samples) {
    MatrixType matrix = create_matrix(n, k);
    DenseVector vector = create_vector(n);
    DenseVector result(n);

    return execute_and_measure([&] {
        result = matrix*vector;
    }, samples);
}

template<typename MatrixType>
double matrix_matrix_multiplication(int n, int k, int samples) {
    MatrixType matrix1 = create_matrix(n, k);
    MatrixType matrix2 = create_matrix(n, k);
    MatrixType result(n, n);

    return execute_and_measure([&] {
        result = matrix1*matrix2;
    }, samples);
}

template<typename MatrixType, typename DecompType>
double solve_linear_system(int n, int k, int samples) {
    MatrixType matrix = create_matrix(n, k);
    DenseVector vector = create_vector(n);
    DenseVector result(n);

    DecompType decomp;

    return execute_and_measure([&] {
        decomp.compute(matrix);
        result = decomp.solve(vector);
    }, samples);
}

int main() {
    srand(0);

    int samples = 1000;

    int k = 5;
    int n_min = 5;
    int n_max = 300;

    std::cout << "Matrix-vector multiplication\n";
    std::cout << "----------------------------\n";
    std::cout << "Dimension\t" << "Dense (Threads: 1)\t" << "Dense (Threads: 2)\t" << "Dense (Threads: 3)\t" << "Dense (Threads: 4)\t" << "Sparse\t" << std::endl;

    for(int n = n_min; n <= n_max; n += 5) {
        std::cout << n << "\t";

        for(int t = 1; t <= 4; ++t) {
            Eigen::setNbThreads(t);
            std::cout << matrix_vector_multiplication<DenseMatrix>(n, k, samples) << "\t";

        }

        Eigen::setNbThreads(1);
        std::cout << matrix_vector_multiplication<SparseMatrix>(n, k, samples) << "\t";
        std::cout << std::endl;
    }

    std::cout << "Matrix-matrix multiplication\n";
    std::cout << "----------------------------\n";
    std::cout << "Dimension\t" << "Dense (Threads: 1)\t" << "Dense (Threads: 2)\t" << "Dense (Threads: 3)\t" << "Dense (Threads: 4)\t" << "Sparse\t" << std::endl;

    for(int n = n_min; n <= n_max; n += 5) {
        std::cout << n << "\t";

        for(int t = 1; t <= 4; ++t) {
            Eigen::setNbThreads(t);
            std::cout << matrix_matrix_multiplication<DenseMatrix>(n, k, samples) << "\t";

        }

        Eigen::setNbThreads(1);
        std::cout << matrix_matrix_multiplication<SparseMatrix>(n, k, samples) << "\t";
        std::cout << std::endl;
    }

    std::cout << "Decomposition and solving\n";
    std::cout << "-------------------------\n";
    std::cout << "Dimension\t" << "Dense: PartialPivLU\t" << "Dense: FullPivLU\t" << "Dense: HouseholderQR\t" << "Dense: CompleteOrthogonalDecomposition\t" << "Dense: LLT\t" << "Dense: LDLT\t" << "Sparse: SimplicialLLT\t" << "Sparse: SimplicialLDLT\t" << "Sparse: SparseLU\t" << "Sparse: ConjugateGradient\t" << "Sparse: LeastSquaresConjugateGradient\t" << "Sparse: BiCGSTAB" << "\t" << std::endl;

    // Dense decompositions, http://eigen.tuxfamily.org/dox/group__TopicLinearAlgebraDecompositions.html
    // Sparse decompositions, http://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html
    Eigen::setNbThreads(1);
    for(int n = n_min; n <= n_max; n += 5) {
        std::cout << n << "\t";
        std::cout << solve_linear_system<DenseMatrix, Eigen::PartialPivLU<DenseMatrix>>(n, k, samples) << "\t";
        std::cout << solve_linear_system<DenseMatrix, Eigen::FullPivLU<DenseMatrix>>(n, k, samples) << "\t";
        std::cout << solve_linear_system<DenseMatrix, Eigen::HouseholderQR<DenseMatrix>>(n, k, samples) << "\t";
        std::cout << solve_linear_system<DenseMatrix, Eigen::CompleteOrthogonalDecomposition<DenseMatrix>>(n, k, samples) << "\t";
        std::cout << solve_linear_system<DenseMatrix, Eigen::LLT<DenseMatrix>>(n, k, samples) << "\t";
        std::cout << solve_linear_system<DenseMatrix, Eigen::LDLT<DenseMatrix>>(n, k, samples) << "\t";
        std::cout << solve_linear_system<SparseMatrix, Eigen::SimplicialLLT<SparseMatrix>>(n, k, samples) << "\t";
        std::cout << solve_linear_system<SparseMatrix, Eigen::SimplicialLDLT<SparseMatrix>>(n, k, samples) << "\t";
        std::cout << solve_linear_system<SparseMatrix, Eigen::SparseLU<SparseMatrix>>(n, k, samples) << "\t";
        //std::cout << solve_linear_system<SparseMatrix, Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>>>(n, k, samples);    // Crashes for some reason
        std::cout << solve_linear_system<SparseMatrix, Eigen::ConjugateGradient<SparseMatrix>>(n, k, samples) << "\t";
        std::cout << solve_linear_system<SparseMatrix, Eigen::LeastSquaresConjugateGradient<SparseMatrix>>(n, k, samples) << "\t";
        std::cout << solve_linear_system<SparseMatrix, Eigen::BiCGSTAB<SparseMatrix>>(n, k, samples) << "\t";
        std::cout << std::endl;
    }
}
