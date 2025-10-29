//
// Created by alfonso on 14/09/21.
//
#include "../distlocRutaCUDA.h"
#include "../distlocRuta.h"


void relacionaLoc2Nodo(PLocalidad ploc, PNodo pnodo, unsigned int tam_loc, unsigned int tam_nodo);



/**
 *
 * @param d_pnodo
 * @param d_ploc
 * @param d_presu
 * @param tam_loc
 * @param tam_n0
 * @param tam_n1
 */
__global__ void
calculadistL2Nv2(PDNodo d_pnodo, PDLocalidad d_ploc, PDResu d_presu, unsigned int tam_loc, unsigned int tam_n0,
                 unsigned int tam_n1) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double dmin = 8;
    long jmin = -1;
    __shared__ double daux;

    if (idx >= tam_loc)return;
    PDLocalidad p = d_ploc + idx;

    if (tam_n0 > 0) {
        dmin = (d_presu + idx)->dist;
        jmin = (d_presu + idx)->index;
    }

    for (unsigned int j = tam_n0; j < tam_n1; j++) {
        PDNodo pn = d_pnodo + j;
        daux = pn->x * p->x + pn->y * p->y + pn->z * p->z;
        if (daux > 1.00)daux = 1.00;
        if (daux < -1.00)daux = -1.00;
        daux = acos(daux);
        if (dmin > daux) {
            dmin = daux;
            jmin = j;
        }
    }

    (d_presu + idx)->dist = dmin;
    (d_presu + idx)->index = jmin;

}




/**
 *
 * @param ploc
 * @param pnodo
 * @param tam_loc
 * @param tam_nodo
 */
void relacionaLoc2Nodo(PLocalidad ploc, PNodo pnodo, unsigned int tam_loc, unsigned int tam_nodo) {

    std::cout << "Relaciona Localidades con Nodos ..." << std::endl;


    int canti_hilos = 1000;
    int canti_bloques = (int) ceil((double) tam_loc / canti_hilos) + 1;


    unsigned int delta = 200000;
    unsigned int tam_n1;

    std::cout << "Hilos: " << canti_hilos << " Bloques: " << canti_bloques << std::endl;

    PDNodo d_pnodo = nullptr;
    PDLocalidad d_ploc = nullptr;
    PDResu d_presu = nullptr;

    auto h_pnodo = (PDNodo) malloc(sizeof(struct DNodo) * tam_nodo);
    auto h_ploc = (PDLocalidad) malloc(sizeof(struct DLocalidad) * tam_loc);
    auto h_presu = (PDResu) malloc(sizeof(struct DResu) * tam_loc);

    for (auto i = 0; i < tam_loc; i++) {
        (h_ploc + i)->x = (ploc + i)->x;
        (h_ploc + i)->y = (ploc + i)->y;
        (h_ploc + i)->z = (ploc + i)->z;
    }

    for (auto i = 0; i < tam_nodo; i++) {
        (h_pnodo + i)->x = (pnodo + i)->x;
        (h_pnodo + i)->y = (pnodo + i)->y;
        (h_pnodo + i)->z = (pnodo + i)->z;
    }

    cudaMalloc((void **) &(d_ploc), tam_loc * sizeof(struct DLocalidad));
    cudaMalloc((void **) &(d_pnodo), tam_nodo * sizeof(struct DNodo));
    cudaMalloc((void **) &(d_presu), tam_loc * sizeof(struct DResu));

    cudaMemcpy(d_ploc, h_ploc, tam_loc * sizeof(struct DLocalidad), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pnodo, h_pnodo, tam_nodo * sizeof(struct DNodo), cudaMemcpyHostToDevice);


    int ii = 1;
    for (unsigned int tam_n0 = 0; tam_n0 < tam_nodo; tam_n0 += delta) {
        tam_n1 = (tam_n0 + delta > tam_nodo) ? tam_nodo : tam_n0 + delta;
        std::cout << "ciclo: " << ii << " " << tam_n0 << " " << tam_n1 << std::endl;
        ii++;
        calculadistL2Nv2<<<canti_bloques, canti_hilos>>>(d_pnodo, d_ploc, d_presu, tam_loc, tam_n0, tam_n1);
        cudaDeviceSynchronize();
    }


    cudaMemcpy(h_presu, d_presu, tam_loc * sizeof(struct DResu), cudaMemcpyDeviceToHost);

    for (auto i = 0; i < tam_loc; i++) {
        (ploc + i)->dist_nodo = (h_presu + i)->dist;
        (ploc + i)->index_nodo = (h_presu + i)->index;
        (ploc + i)->id_nodo = (pnodo + (ploc + i)->index_nodo)->id_nodo;
    }

    cudaFree(d_presu);
    cudaFree(d_pnodo);
    cudaFree(d_ploc);

    free(h_presu);
    free(h_ploc);
    free(h_pnodo);
}