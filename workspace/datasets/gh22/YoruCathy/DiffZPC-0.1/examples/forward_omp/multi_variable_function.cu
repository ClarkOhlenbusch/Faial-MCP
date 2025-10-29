// TODO: need a tuple

#include <cuda_runtime.h>
#include <iostream>
#include "zensim/cuda/memory/MemOps.hpp"
#include "zensim/cuda/Cuda.h"

#include "zensim/container/Vector.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/types/Tuple.h"
#include "zensim/math/matrix/Matrix.hpp"

#include "zensim/autodiff/forward/dual.hpp"
using namespace autodiff;

// The multi-variable function for which derivatives are needed
constexpr dual f(dual x, dual y, dual z)
{
  return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

int main() {
  using namespace zs;
  fmt::print("\n\n==========example-forward-single-variable-function===========\n\n");
  auto ompPol=omp_exec().profile(true);
  //  Vector<float> val{3, memsrc_e::host};
  //  tuple<Vector<float>> vals;
  Vector<tuple<autodiff::dual, autodiff::dual, autodiff::dual>> val{1, memsrc_e::host};
  val[0] = {1.0, 2.0, 3.0};
  //



  //  for(int i=0; i!=val.size();++i){
  //    std::cout<<"val[i]"<<val[i]<<std::endl;
  //  }

  ompPol(Collapse{val.size()}, [val = proxy<execspace_e::openmp>(val)](int i) mutable {
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
  //    dual x = 1.0;
  //    dual y = 2.0;
  //    dual z = 3.0;
  //
  //    dual u = f(x, y, z);
  //
  //    double dudx = derivative(f, wrt(x), at(x, y, z));
  //    double dudy = derivative(f, wrt(y), at(x, y, z));
  //    double dudz = derivative(f, wrt(z), at(x, y, z));
  //
  //    std::cout << "u = " << u << std::endl;         // print the evaluated output u = f(x, y, z)
  //    std::cout << "du/dx = " << dudx << std::endl;  // print the evaluated derivative du/dx
  //    std::cout << "du/dy = " << dudy << std::endl;  // print the evaluated derivative du/dy
  //    std::cout << "du/dz = " << dudz << std::endl;  // print the evaluated derivative du/dz

  return 0;
}