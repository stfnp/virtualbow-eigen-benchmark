/*
#include <iostream>
#include <Eigen/Sparse>

int main() {
    Eigen::SparseMatrix<double> matrix(10, 10);
    matrix.coeffRef(0, 0) = 1.0;

    std::cout << matrix.coeffs().transpose() << "\n";

    matrix.coeffRef(3, 3) = 2.0;
    matrix.makeCompressed();

    std::cout << matrix.coeffs().transpose() << "\n";

    matrix.coeffRef(2, 2) = 3.0;
    matrix.makeCompressed();

    std::cout << matrix.coeffs().transpose() << "\n";

}
*/

/*
#include "utils.hpp"

#include <iostream>
#include <Eigen/Sparse>
#include <random>
#include <fstream>
#include <chrono>

using namespace Eigen;

typedef Eigen::Triplet<double> T;


void findDuplicates(std::vector<std::pair<int, int>>& dummypair, Ref<VectorXi> multiplicity) {
    // Iterate over the vector and store the frequency of each element in map
    int pairCount = 0;
    std::pair<int, int> currentPair;
    for (int i = 0; i < multiplicity.size(); ++i) {
        currentPair = dummypair[pairCount];
        while (currentPair == dummypair[pairCount + multiplicity[i]]) {
            multiplicity[i]++;
        }
        pairCount += multiplicity[i];
    }
}

int main() {
    //init random generators
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    int rows = 1000;
    int cols = rows;
    std::uniform_int_distribution<int> distentryrow(0, rows-1);
    std::uniform_int_distribution<int> distentrycol(0, cols-1);

    std::vector<T> tripletList;
    SparseMatrix<double> matrix(rows, cols);

    //generate sparsity pattern of matrix with  10% fill-in
    tripletList.emplace_back(3, 0, 15);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            auto value = dist(gen);                          //generate random number
            auto value2 = dist(gen);                         //generate random number
            auto value3 = dist(gen);                         //generate random number
            if (value < 0.05) {
                auto rowindex = distentryrow(gen);
                auto colindex = distentrycol(gen);
                tripletList.emplace_back(rowindex, colindex, value);      //if larger than treshold, insert it

                //dublicate every third entry to mimic entries which appear more then once
                if (value2 < 0.3333333333333333333333)
                    tripletList.emplace_back(rowindex, colindex, value);

                //triple every forth entry to mimic entries which appear more then once
                if (value3 < 0.25)
                    tripletList.emplace_back(rowindex, colindex, value);
            }
        }
    tripletList.emplace_back(3, 0, 9);

    int numberOfValues = tripletList.size();

    //initially set all matrices from triplet to allocate space and sparsity pattern
    matrix.setFromTriplets(tripletList.begin(), tripletList.end());

    int nnz = matrix.nonZeros();
    //reset all entries back to zero to fill in later
    matrix.coeffs().setZero();

    //document sorting of entries for repetitive insertion
    VectorXi internalIndex(numberOfValues);
    std::vector<std::pair<int, int>> dummypair(numberOfValues);

    VectorXd valuelist(numberOfValues);
    for(int l = 0; l < numberOfValues; ++l) {
        valuelist(l) = tripletList[l].value();
    }

    //init internalindex and dummy pair
    internalIndex = Eigen::VectorXi::LinSpaced(numberOfValues, 0.0, numberOfValues - 1);
    for (int i = 0; i < numberOfValues; ++i) {
        dummypair[i].first = tripletList[i].col();
        dummypair[i].second = tripletList[i].row();
    }

    // sort the vector  internalIndex based on the dummypair
    std::sort(internalIndex.begin(), internalIndex.end(), [&](int i, int j) {
        return dummypair[i].first < dummypair[j].first || (dummypair[i].first == dummypair[j].first && dummypair[i].second < dummypair[j].second);
    });

    VectorXi dublicatecount(nnz);
    dublicatecount.setOnes();
    findDuplicates(dummypair, dublicatecount);

    dummypair.clear();

    //calculate vector containing all indices of triplet
    //therefore vector[k] is the vectorXi containing the entries of triples which should be written at dof k
    int indextriplet = 0;
    int multiplicity = 0;

    std::vector<VectorXi> listofentires(matrix.nonZeros());
    for (int k = 0; k < matrix.nonZeros(); ++k) {
        multiplicity = dublicatecount[k];
        listofentires[k] = internalIndex.segment(indextriplet, multiplicity);
        indextriplet += multiplicity;
    }


    //========================================
    //Here the nonlinear analysis should start and everything beforehand is prepocessing

    int samples = 50;
    double time = execute_and_measure([&]{
        //Test2 use internalIndex but calculate listofentires on the fly

        matrix.coeffs().setZero();

        indextriplet = 0;
        for (int k = 0; k < matrix.nonZeros(); ++k) {
            multiplicity = dublicatecount[k];
            matrix.coeffs()[k] += valuelist(internalIndex.segment(indextriplet, multiplicity)).sum();
            indextriplet += multiplicity;
        }

    }, samples);

    std::cout << "Time: " << time << "\n";

    return 0;
}
*/

#include "utils.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

using Eigen::SparseMatrix;
using Eigen::MatrixXd;
using Eigen::VectorXi;
using Eigen::Triplet;


struct Element {
    MatrixXd matrix;
    VectorXi indices;
};

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
        matrix = SparseMatrix<double>(max_index + 1, max_index + 1);
    }

    void assemble() {
        matrix.setZero();
        for(auto& e: elements) {
            for(int i = 0; i < e.matrix.rows(); ++i) {
                for(int j = 0; j < e.matrix.cols(); ++j) {
                    matrix.coeffRef(e.indices(i), e.indices(j)) += e.matrix(i, j);
                }
            }
        }
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

    void assemble() {
        matrix.setZero();
        triplets.clear();
        for(auto& e: elements) {
            for(int i = 0; i < e.matrix.rows(); ++i) {
                for(int j = 0; j < e.matrix.cols(); ++j) {
                    triplets.emplace_back(e.indices(i), e.indices(j), e.matrix(i, j));
                }
            }
        }
        matrix.setFromTriplets(triplets.begin(), triplets.end());
    }
};

// Pointer method
// For each element, store the pointers of the global matrix entries associated with the local element entries
struct System_Pointers {
    SparseMatrix<double> matrix;
    std::vector<Element> elements;
    std::vector<std::vector<double*>> references;

    System_Pointers(const std::vector<Element>& elements)
        : elements(elements)
    {
        //Determine matrix size
        int max_index = 0;
        for(auto& e: elements) {
            max_index = std::max(max_index, e.indices.maxCoeff());
        }

        // Initialize matrix
        matrix = SparseMatrix<double>(max_index + 1, max_index + 1);

        // Initialize sparsity pattern and store references
        for(auto& e: elements) {
            // TODO: Use iterators here
            for(int i = 0; i < e.matrix.rows(); ++i) {
                for(int j = 0; j < e.matrix.cols(); ++j) {
                    matrix.coeffRef(e.indices(i), e.indices(j));
                }
            }
        }

        // Store references
        for(auto& e: elements) {
            std::vector<double*> r;
            // TODO: Use iterators here
            for(int i = 0; i < e.matrix.rows(); ++i) {
                for(int j = 0; j < e.matrix.cols(); ++j) {
                    r.push_back(&matrix.coeffRef(e.indices(i), e.indices(j)));
                }
            }
            references.push_back(r);
        }
    }

    void assemble() {
        matrix.setZero();
        for(size_t k = 0; k < elements.size(); ++k) {
            auto& e = elements[k];
            auto& r = references[k];

            // TODO: Use iterators here
            for(int i = 0; i < e.matrix.size(); ++i) {
                *r[i] += e.matrix.reshaped()(i);
            }
        }
    }
};

// Somewhat like the pointer method, but storing indices instead
struct System_Indices {
    SparseMatrix<double> matrix;
    std::vector<Element> elements;

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
    }

    void assemble() {
        matrix.data().atInRange(0, 0, 0);
    }
};

// Transformation method
// Use matrix operations to transform element matrices to global coordinates and then just sum them up
// Idea: Also store transposed transform
// Idea: Look at PermutationMatrix
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

    void assemble() {
        matrix.setZero();
        SparseMatrix<double> temp;
        for(int i = 0; i < elements.size(); ++i) {
            temp = elements[i].matrix.sparseView();
            matrix += transforms[i] * temp * transforms_t[i];
        }
    }
};

// Generates a random, symmetric, positive definite matrix with dimension n
// Source: https://math.stackexchange.com/a/358092
MatrixXd random_matrix(int n) {
    MatrixXd A = 0.5*(MatrixXd::Random(n, n) + MatrixXd::Identity(n, n));    // Matrix with elements in [0, 1]
    return 0.5*(A.transpose() + A) + n*MatrixXd::Identity(n, n);    // Construct symmetry and diagonal dominance
}

std::vector<Element> create_elements(int dimension, int n_elements) {
    std::vector<Element> elements;
    for(int i = 0; i < n_elements; ++i) {
        int start_index = i*dimension/2;
        elements.push_back(Element{
            .matrix = random_matrix(dimension),
            .indices = VectorXi::LinSpaced(dimension, start_index, start_index + dimension - 1)
        });
    }
    return elements;
}

// TODO:
// * Better determination of maximum index in the elements, use std algorithm for that

int main() {
    int samples = 100;

    std::vector<Element> elements = create_elements(4, 10);

    /*
    System_CoeffRef system1(elements);
    std::cout  << "CoeffRef: " << execute_and_measure([&] { system1.assemble(); }, samples) << "\n";

    System_Triplets system2(elements);
    std::cout  << "Triplets: " << execute_and_measure([&] { system2.assemble(); }, samples) << "\n";

    System_Pointers system3(elements);
    std::cout  << "Pointers: " << execute_and_measure([&] { system3.assemble(); }, samples) << "\n";

    System_Transform system4(elements);
    std::cout  << "Transform: " << execute_and_measure([&] { system4.assemble(); }, samples) << "\n";
    */

    //System_Indices system5(elements);
    //std::cout  << "Indices: " << execute_and_measure([&] { system5.assemble(); }, samples) << "\n";

    SparseMatrix<double> A(5, 5);
    A.insert(0, 0) = 1.0;
    A.insert(2, 3) = 2.0;
    A.insert(3, 2) = 3.0;
    A.insert(4, 4) = 4.0;
    A.makeCompressed();

    std::cout << A.toDense() << "\n\n";
    std::cout << A.coeffs().transpose() << "\n\n";



    //System_References system(elements);
    //system.assemble();
    //std::cout << system.matrix.toDense() << "\n\n";
}
