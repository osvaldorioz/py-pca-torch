#include <iostream>
#include <Eigen/Dense>

//sudo apt install libeigen3-dev
//g++ -o eigen_test eigen_test.cpp -I /usr/include/eigen3

int main() {
    Eigen::Matrix2d mat;
    mat << 1, 2,
           3, 4;
    std::cout << "Matriz:\n" << mat << std::endl;
    return 0;
}
