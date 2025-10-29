#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define MAX_ROTATIONS  80
#define GENERAL_MEMORY_PROBLEM printf( "You do not have enough memory ([m|re]alloc failure)\nDying\n\n" ) ; exit( EXIT_FAILURE ) ;

struct Atom{
    int charge;
    int name;
};
struct Residue
{
    int n;
    struct Atom *atoms;

};



int main()
{
    struct Residue r[2];


    cudaMalloc((void**)&r[0].atoms,2*sizeof(struct Atom));
    cudaDeviceSynchronize();
    cudaMalloc((void**)&r[1].atoms,2*sizeof(struct Atom));
    cudaDeviceSynchronize();
    printf("hello whatsup");


    

}

