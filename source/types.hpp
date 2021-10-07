#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

using Real = double;

using SparseMatrix = Eigen::SparseMatrix<Real>;
using DenseMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using DenseVector = Eigen::Vector<Real, Eigen::Dynamic>;
using IndexVector = Eigen::VectorXi;
using Triplet = Eigen::Triplet<Real>;
