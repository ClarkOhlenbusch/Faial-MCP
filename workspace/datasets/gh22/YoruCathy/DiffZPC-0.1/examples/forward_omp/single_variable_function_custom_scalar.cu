// TODO: this does not work

//#include <cuda_runtime.h>
//#include <iostream>
//#include <complex>
//using namespace std;
//
//#include "zensim/autodiff/forward/dual.hpp"
//#include "zensim/cuda/memory/MemOps.hpp"
//#include "zensim/cuda/Cuda.h"
//#include "zensim/container/Vector.hpp"
//#include "zensim/omp/execution/ExecutionPolicy.hpp"
//
//using namespace autodiff;
//// Specialize isArithmetic for complex to make it compatible with dual
//namespace autodiff::detail {
//
//    template<typename T>
//    struct ArithmeticTraits<complex<T>> : ArithmeticTraits<T> {};
//
//} // autodiff::detail
//
//using cxdual = Dual<complex<double>, complex<double>>;
//
//// The single-variable function for which derivatives are needed
//cxdual f(cxdual x)
//{
//    return log(x) + 1.0 + x + x*x + 1/x;
//}
//
//int main() {
//  using namespace zs;
//  fmt::print("\n\n==========example-forward-single-variable-function-custom-scalar===========\n\n");
//  auto ompPol=omp_exec().profile(true);
//  Vector<cxdual> val{1, memsrc_e::host};
//  Vector<cxdual> val1{1, memsrc_e::host};
//  for(int i=0; i!=val.size();++i){
//    val[i]=10.0;
//  }
//  for(int i=0; i!=val.size();++i){
//    std::cout<<"val[i]"<<val[i]<<std::endl;
//  }
//
//  ompPol(zip(val,val1), [](auto &val, auto &val1) mutable {
////    printf("i:%f\n", i);
////    printf("val[i]: (%f)\n", val[i]);
//    printf("val: %f\n", val.val);
//    //    printf("val: %f\n", val.grad);
//    auto tmp = derivative(f, wrt(val), at(val));;
//    printf("val.grad: (%f)\n", val.grad);
//    printf("val.grad2: (%f)\n", tmp);
//  });
//
//
///* -----old example ------*/
////  autodiff::dual x = 2.0;                                 // the input variable x
////  autodiff::dual u = f(x);                                // the output variable u
////
////  double dudx = derivative(f, wrt(x), at(x));   // evaluate the derivative du/dx
////
////  std::cout << "u = " << u << std::endl;        // print the evaluated output u
////  std::cout << "du/dx = " << dudx << std::endl; // print the evaluated derivative du/dx
//
//  return 0;
//}

// C++ includes
#include <iostream>
#include <complex>
using namespace std;

// autodiff include
#include <zensim/autodiff/forward/dual.hpp>
using namespace autodiff;

// Specialize isArithmetic for complex to make it compatible with dual
namespace autodiff::detail {

    template<typename T>
    struct ArithmeticTraits<complex<T>> : ArithmeticTraits<T> {};

} // autodiff::detail

using cxdual = Dual<complex<double>, complex<double>>;

// The single-variable function for which derivatives are needed
cxdual f(cxdual x)
{
    return 1 + x + x*x + 1/x + log(x);
}

int main()
{
    cxdual x = 2.0;   // the input variable x
    cxdual u = f(x);  // the output variable u

    cxdual dudx = derivative(f, wrt(x), at(x));  // evaluate the derivative du/dx

    cout << "u = " << u << endl;         // print the evaluated output u
    cout << "du/dx = " << dudx << endl;  // print the evaluated derivative du/dx
}
