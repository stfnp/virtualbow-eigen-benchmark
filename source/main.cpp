#include "utils.hpp"
#include "assembly.hpp"
#include "operations.hpp"
#include <iostream>

int main() {
    int samples = 100;

    int n_dimension = 8;
    int n_elements_min = 1;
    int n_elements_max = 100;

    std::cout << "Sparse matrix update\n";
    std::cout << "Dimension\t" << "CoeffRef\t" << "Triplets\t" << "Transform\t" << "Pointers\t" << "Indices\t" << "\n";
    for(int n_elements = n_elements_min; n_elements <= n_elements_max; n_elements += 1) {
        std::vector<Element> elements = create_elements(n_dimension, n_elements);
        std::cout << n_dimension*(n_elements + 1)/2 << "\t";

        System_CoeffRef system1(elements);
        std::cout << execute_and_measure([&]{ system1.assemble(); }, samples) << "\t";

        System_Triplets system2(elements);
        std::cout << execute_and_measure([&]{ system2.assemble(); }, samples) << "\t";

        System_Transform system3(elements);
        std::cout << execute_and_measure([&]{ system3.assemble(); }, samples) << "\t";

        System_Pointers system4(elements);
        std::cout << execute_and_measure([&]{ system4.assemble(); }, samples) << "\t";

        System_Indices system5(elements);
        std::cout << execute_and_measure([&]{ system5.assemble(); }, samples) << "\t";
        std::cout << std::endl;
    }
    std::cout << "\n";

    std::cout << "Matrix-vector multiplication\n";
    std::cout << "Dimension\t" << "Dense (Threads: 1)\t" << "Dense (Threads: 2)\t" << "Dense (Threads: 3)\t" << "Dense (Threads: 4)\t" << "Sparse\t" << std::endl;
    for(int n_elements = n_elements_min; n_elements <= n_elements_max; n_elements += 1) {
        std::cout << n_dimension*(n_elements + 1)/2 << "\t";

        for(int t = 1; t <= 4; ++t) {
            Eigen::setNbThreads(t);
            std::cout << matrix_vector_multiplication<MatrixXd>(n_elements, n_dimension, samples) << "\t";
        }

        Eigen::setNbThreads(1);
        std::cout << matrix_vector_multiplication<SparseMatrix<double>>(n_elements, n_dimension, samples) << "\t";
        std::cout << std::endl;
    }
    std::cout << "\n";

    std::cout << "Matrix-matrix multiplication\n";
    std::cout << "Dimension\t" << "Dense (Threads: 1)\t" << "Dense (Threads: 2)\t" << "Dense (Threads: 3)\t" << "Dense (Threads: 4)\t" << "Sparse\t" << std::endl;
    for(int n_elements = n_elements_min; n_elements <= n_elements_max; n_elements += 1) {
        std::cout << n_dimension*(n_elements + 1)/2 << "\t";

        for(int t = 1; t <= 4; ++t) {
            Eigen::setNbThreads(t);
            std::cout << matrix_matrix_multiplication<MatrixXd>(n_elements, n_dimension, samples) << "\t";
        }

        Eigen::setNbThreads(1);
        std::cout << matrix_matrix_multiplication<SparseMatrix<double>>(n_elements, n_dimension, samples) << "\t";
        std::cout << std::endl;
    }
    std::cout << "\n";

    std::cout << "Solving a linear system\n";
    std::cout << "Dimension\t" << "Dense: PartialPivLU\t" << "Dense: FullPivLU\t" << "Dense: HouseholderQR\t" << "Dense: CompleteOrthogonalDecomposition\t" << "Dense: LLT\t" << "Dense: LDLT\t" << "Sparse: SimplicialLLT\t" << "Sparse: SimplicialLDLT\t" << "Sparse: SparseLU\t" << "Sparse: ConjugateGradient\t" << "Sparse: LeastSquaresConjugateGradient\t" << "Sparse: BiCGSTAB" << "\t" << std::endl;
    Eigen::setNbThreads(2);
    for(int n_elements = n_elements_min; n_elements <= n_elements_max; n_elements += 1) {
        std::cout << n_dimension*(n_elements + 1)/2 << "\t";
        // Dense decompositions, http://eigen.tuxfamily.org/dox/group__TopicLinearAlgebraDecompositions.html
        // Sparse decompositions, http://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html
        std::cout << solve_linear_system<MatrixXd, Eigen::PartialPivLU<MatrixXd>>(n_elements, n_dimension, samples) << "\t";
        std::cout << solve_linear_system<MatrixXd, Eigen::FullPivLU<MatrixXd>>(n_elements, n_dimension, samples) << "\t";
        std::cout << solve_linear_system<MatrixXd, Eigen::HouseholderQR<MatrixXd>>(n_elements, n_dimension, samples) << "\t";
        std::cout << solve_linear_system<MatrixXd, Eigen::CompleteOrthogonalDecomposition<MatrixXd>>(n_elements, n_dimension, samples) << "\t";
        std::cout << solve_linear_system<MatrixXd, Eigen::LLT<MatrixXd>>(n_elements, n_dimension, samples) << "\t";
        std::cout << solve_linear_system<MatrixXd, Eigen::LDLT<MatrixXd>>(n_elements, n_dimension, samples) << "\t";
        std::cout << solve_linear_system<SparseMatrix<double>, Eigen::SimplicialLLT<SparseMatrix<double>>>(n_elements, n_dimension, samples) << "\t";
        std::cout << solve_linear_system<SparseMatrix<double>, Eigen::SimplicialLDLT<SparseMatrix<double>>>(n_elements, n_dimension, samples) << "\t";
        std::cout << solve_linear_system<SparseMatrix<double>, Eigen::SparseLU<SparseMatrix<double>>>(n_elements, n_dimension, samples) << "\t";
        //std::cout << solve_linear_system<SparseMatrix<double>, Eigen::SparseQR<SparseMatrix<double>, Eigen::COLAMDOrdering<int>>>(n_elements, n_dimension, samples);    // Crashes for some reason
        std::cout << solve_linear_system<SparseMatrix<double>, Eigen::ConjugateGradient<SparseMatrix<double>>>(n_elements, n_dimension, samples) << "\t";
        std::cout << solve_linear_system<SparseMatrix<double>, Eigen::LeastSquaresConjugateGradient<SparseMatrix<double>>>(n_elements, n_dimension, samples) << "\t";
        std::cout << solve_linear_system<SparseMatrix<double>, Eigen::BiCGSTAB<SparseMatrix<double>>>(n_elements, n_dimension, samples) << "\t";
        std::cout << std::endl;
    }
}
