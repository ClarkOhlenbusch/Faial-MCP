#include <iterator>

#include <thrust/universal_vector.h>

#include "irregularia/device/hw.cuh"

auto hello_world_device() -> void
{
  auto letters = "Hello World";
  thrust::universal_vector<char> v {letters,
                                    letters + std::size("Hello World") - 1};
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<char>(std::cout, " "));
}