#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <Eigen/Dense>

namespace py = pybind11;

torch::Tensor pca(torch::Tensor data, int64_t num_components) {
    // Verificar que el tensor de entrada sea de tipo float
    if (data.dtype() != torch::kFloat32) {
        data = data.to(torch::kFloat32);
    }

    // Convertir el tensor de entrada a una matriz de Eigen
    auto data_accessor = data.accessor<float, 2>();
    Eigen::MatrixXf data_matrix(data.size(0), data.size(1));
    for (int64_t i = 0; i < data.size(0); ++i) {
        for (int64_t j = 0; j < data.size(1); ++j) {
            data_matrix(i, j) = data_accessor[i][j];
        }
    }

    // Centrar los datos restando la media
    Eigen::VectorXf mean = data_matrix.colwise().mean();
    Eigen::MatrixXf centered = data_matrix.rowwise() - mean.transpose();

    // Calcular la matriz de covarianza
    Eigen::MatrixXf cov = (centered.adjoint() * centered) / float(data_matrix.rows() - 1);

    // Realizar la descomposición en valores propios
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(cov);
    if (eigensolver.info() != Eigen::Success) {
        throw std::runtime_error("La descomposición en valores propios falló.");
    }
    Eigen::VectorXf eigenvalues = eigensolver.eigenvalues();
    Eigen::MatrixXf eigenvectors = eigensolver.eigenvectors();

    // Seleccionar los 'num_components' vectores propios principales
    Eigen::MatrixXf selected_vectors = eigenvectors.rightCols(num_components);

    // Proyectar los datos originales en el nuevo espacio reducido
    Eigen::MatrixXf transformed_data = centered * selected_vectors;

    // Convertir la matriz de Eigen de vuelta a un tensor de PyTorch
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor result = torch::from_blob(transformed_data.data(), {data.size(0), num_components}, options).clone();

    return result;
}

PYBIND11_MODULE(pca_extension, m) {
    m.def("pca", &pca, "Análisis de Componentes Principales (PCA)",
          py::arg("data"), py::arg("num_components"));
}
