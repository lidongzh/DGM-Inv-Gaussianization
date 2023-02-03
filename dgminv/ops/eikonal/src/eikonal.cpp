#include <stdio.h>
#include <torch/extension.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "omp.h"
#include "eikonal.h"
using namespace std;

torch::Tensor eikonal_forward(const torch::Tensor &f_tensor,
                              int srcx,
                              int srcz,
                              double h) {
  int m = f_tensor.size(1) - 1; // width
  int n = f_tensor.size(0) - 1; // depth
  srcz = srcz;
  srcx = srcx;
  assert(srcz >= 0);
  assert(srcz <= n);
  assert(srcx >= 0);
  assert(srcx <= m);
  // auto u_tensor = torch::zeros_like(f_tensor);
  auto u_tensor = torch::zeros({f_tensor.size(0), f_tensor.size(1)}, torch::dtype(torch::kFloat64));
  auto u = u_tensor.data_ptr<double>();
  auto f = f_tensor.data_ptr<double>();
  forward(u, f, m, n, h, srcx, srcz);
  return u_tensor;
}


std::vector<torch::Tensor> eikonal_backward(const torch::Tensor &grad_u_tensor,
                                        const torch::Tensor &u_tensor,
                                        const torch::Tensor &f_tensor,
                                        int srcx,
                                        int srcz,
                                        double h) {
  int m = f_tensor.size(1) - 1;
  int n = f_tensor.size(0) - 1;
  srcz = srcz;
  srcx = srcx;
  // auto grad_f_tensor = torch::zeros_like(u_tensor);
  auto grad_f_tensor = torch::zeros({f_tensor.size(0), f_tensor.size(1)}, torch::dtype(torch::kFloat64));
  auto grad_f = grad_f_tensor.data_ptr<double>();
  auto grad_u = grad_u_tensor.data_ptr<double>();
  auto u = u_tensor.data_ptr<double>();
  auto f = f_tensor.data_ptr<double>();
  backward(grad_f, grad_u, u, f, m, n, h, srcx, srcz);
  return {grad_f_tensor};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &eikonal_forward, "forward");
  m.def("backward", &eikonal_backward, "backward");
}