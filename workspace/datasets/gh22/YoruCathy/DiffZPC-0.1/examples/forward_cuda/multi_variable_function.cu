#include <cuda_runtime.h>
#include <iostream>
#include "zensim/cuda/memory/MemOps.hpp"
#include "zensim/cuda/Cuda.h"

#include "zensim/container/Vector.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"

#include "zensim/autodiff/forward/dual.hpp"
//using namespace autodiff;

constexpr autodiff::dual f(autodiff::dual x, autodiff::dual y, autodiff::dual z)
{
  auto tmp = autodiff::dual::def(1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z));
  return tmp;
}

int main() {
  using namespace zs;
  //  fmt::print("\n\n==========example-forward-single-variable-function===========\n\n");
  auto cudaPol=cuda_exec().profile(true);
  //  Vector<float> val{3, memsrc_e::host};
  //  tuple<Vector<float>> vals;
  Vector<tuple<autodiff::dual, autodiff::dual, autodiff::dual>> val{1, memsrc_e::device, 0};
  //  val[0] = {1.0, 2.0, 3.0};
  //

  cudaPol(enumerate(val), [] __device__(auto id, auto &v) mutable {
    v = {1.0, 2.0, 3.0};
    //    printf("v: %f", (float)v);
  });

  //  for(int i=0; i!=val.size();++i){
  //    std::cout<<"val[i]"<<val[i]<<std::endl;
  //  }

  cudaPol(Collapse{val.size()}, [val = proxy<execspace_e::cuda>(val)]__device__(int i) mutable {
    auto xyz = val[i];
    autodiff::dual x = get<0>(xyz);
    autodiff::dual y = get<1>(xyz);
    autodiff::dual z = get<2>(xyz);
    //    double dudx = derivative(f, wrt(x), at(x, y, z));
    double dudx = derivative(f, wrt(x), at(x, y, z));
    double dudy = derivative(f, wrt(y), at(x, y, z));
    double dudz = derivative(f, wrt(z), at(x, y, z));

    printf("x.grad: (%f)\n", dudx);
    printf("y.grad: (%f)\n", dudy);
    printf("z.grad: (%f)\n", dudz);
  });
  /* -----old example ------*/
  //  autodiff::dual x = 2.0;                                 // the input variable x
  //  autodiff::dual u = f(x);                                // the output variable u
  //
  //  double dudx = derivative(f, wrt(x), at(x));   // evaluate the derivative du/dx
  //
  //  std::cout << "u = " << u << std::endl;        // print the evaluated output u
  //  std::cout << "du/dx = " << dudx << std::endl; // print the evaluated derivative du/dx

  return 0;
}