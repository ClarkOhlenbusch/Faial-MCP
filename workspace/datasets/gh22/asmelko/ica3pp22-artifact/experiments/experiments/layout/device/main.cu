#include "matrix_multiply/matrix_multiply.cuh"
#include "matrix_transpose/matrix_transpose.cuh"


int main(int argc, char** argv)
{
	std::vector<std::string> args(argv + 1, argv + argc);

	{
		std::cout << "<<<<<multiply>>>>>" << std::endl;
		matrix_multiply_experiment e(args);
		e.run();
	}
}