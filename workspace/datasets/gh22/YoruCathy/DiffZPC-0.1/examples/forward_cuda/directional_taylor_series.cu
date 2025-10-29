// TODO: need a Tuple

#include <cuda_runtime.h>
#include <iostream>
#include "zensim/cuda/memory/MemOps.hpp"
#include "zensim/cuda/Cuda.h"

#include "zensim/container/Vector.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

// autodiff include
#include <zensim/autodiff/forward/real.hpp>
using namespace autodiff;

// The scalar function for which a 4th order directional Taylor series will be computed.
real4th f(const real4th& x, const real4th& y, const real4th& z)
{
    return sin(x * y) * cos(x * z) * exp(z);
}

int main()
{
//    using namespace zs;
//    fmt::print("\n\n==========example-forward-single-variable-function===========\n\n");
//    auto ompPol=omp_exec().profile(true);
//    Vector<autodiff::dual> val{1, memsrc_e::host};
//    Vector<autodiff::dual> val1{1, memsrc_e::host};
//    for(int i=0; i!=val.size();++i){
//        val[i]=10.0;
//    }
//    for(int i=0; i!=val.size();++i){
//        std::cout<<"val[i]"<<val[i]<<std::endl;
//    }
//
//    ompPol(zip(val,val1), [](auto &val, auto &val1) mutable {
//        printf("val: %f\n", val.val);
//        auto tmp = derivative(f, wrt(val), at(val));;
//        printf("val.grad: (%f)\n", val.grad);
//        printf("val.grad2: (%f)\n", tmp);
//    });


    real4th x = 1.0;                                       // the input vector x
    real4th y = 2.0;                                       // the input vector y
    real4th z = 3.0;                                       // the input vector z

    auto g = taylorseries(f, along(1, 1, 2), at(x, y, z)); // the function g(t) as a 4th order Taylor approximation of f(x + t, y + t, z + 2t)

    double t = 0.1;                                        // the step length used to evaluate g(t), the Taylor approximation of f(x + t, y + t, z + 2t)

    real4th u = f(x + t, y + t, z + 2*t);                  // the exact value of f(x + t, y + t, z + 2t)

    double utaylor = g(t);                                 // the 4th order Taylor estimate of f(x + t, y + t, z + 2t)

    std::cout << std::fixed;
    std::cout << "Comparison between exact evaluation and 4th order Taylor estimate of f(x + t, y + t, z + 2t):" << std::endl;
    std::cout << "u(exact)  = " << u << std::endl;
    std::cout << "u(taylor) = " << utaylor << std::endl;
}

/*-------------------------------------------------------------------------------------------------
=== Output ===
---------------------------------------------------------------------------------------------------
Comparison between exact evaluation and 4th order Taylor estimate of f(x + t, y + t, z + 2t):
u(exact)  = -16.847071
u(taylor) = -16.793986
-------------------------------------------------------------------------------------------------*/
