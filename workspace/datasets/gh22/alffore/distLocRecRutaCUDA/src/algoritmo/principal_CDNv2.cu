//
// Created by alfonso on 25/09/21.
//

#include "../distlocRutaCUDA.h"

void generaRutas(PNodo pnodo, unsigned int tam_nodo);

extern __global__ void
kernel_principalDN(PNodoRed d_pnodored, PSubNodo d_psubnodo, PResRec d_presrec, unsigned int tam_nodo);

extern double sumaVariacion(PResRec d_presrec, unsigned int tam_nodo);




/**
 * Kernel que recupera los indices para los identificadores de los subnodos
 * @param d_pnodored
 * @param d_psubnodo
 * @param d_pidnodo
 * @param tam_nodo
 */
__global__ void
kernel_indexaNodos(PNodoRed d_pnodored, PSubNodo d_psubnodo, const long *d_pidnodo, unsigned int tam_nodo,
                   unsigned int tam_n0, unsigned int tam_n1) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= tam_nodo)return;

    PNodoRed pnr = d_pnodored + idx;

    for (int i = 0; i < pnr->nsn; i++) {
        for (unsigned int j = tam_n0; j < tam_n1; j++) {
            if (*(d_pidnodo + j) == pnr->aid_nodos[i]) {
                (d_psubnodo + idx)->aindex_nodos[i] = j;
                break;
            }
        }
    }
    (d_psubnodo + idx)->nsn = pnr->nsn;
}


/**
 * Kernel que prepara el arreglo de resultados distancias ids, dist
 * @param d_pnodored
 * @param d_presrec
 * @param tam_nodo
 */
__global__ void kernel_preparaDist(PNodoRed d_pnodored, PResRec d_presrec, unsigned int tam_nodo) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= tam_nodo)return;

    (d_presrec + idx)->id_rec = (d_pnodored + idx)->id_rec;
    (d_presrec + idx)->dist_rec = (d_pnodored + idx)->dist_rec;
    (d_presrec + idx)->delta = 0.0;
}


/**
 *
 * @param pnodo
 * @param tam_nodo
 */
void generaRutas(PNodo pnodo, unsigned int tam_nodo) {

    int canti_hilos = 1000;
    int canti_bloques = (int) ceil((double) tam_nodo / canti_hilos);

    std::cout << "Bloques: " << canti_bloques << " Hilos: " << canti_hilos << std::endl;

    auto h_pnodored = (PNodoRed) malloc(sizeof(struct NodoRed) * tam_nodo);
    auto h_pidnodo = (long *) malloc(sizeof(long) * tam_nodo);
    auto h_psubnodo = (PSubNodo) malloc(sizeof(struct SubNodo) * tam_nodo);
    auto h_presrec = (PResRec) malloc(sizeof(struct ResRec) * tam_nodo);


    PNodoRed d_pnodored;
    long *d_pidnodo;
    PSubNodo d_psubnodo;
    PResRec d_presrec;


    for (unsigned int i = 0; i < tam_nodo; i++) {
        PNodo p = pnodo + i;

        if (p->id_rec > 0) {
            (h_pnodored + i)->id_rec = p->id_rec;
            (h_pnodored + i)->dist_rec = p->dist_rec;
        } else {
            (h_pnodored + i)->id_rec = -1;
            (h_pnodored + i)->dist_rec = MAX_DIST;
        }

        (h_pnodored + i)->nsn = 0;
        if (p->nsn > 0) {
            for (int j = 0; j < p->nsn; j++) {
                (h_pnodored + i)->aid_nodos[j] = p->aid_nodos[j];
                (h_pnodored + i)->adist[j] = p->adist[j];
            }
            (h_pnodored + i)->nsn = p->nsn;
        }

        *(h_pidnodo + i) = p->id_nodo;

    }

    cudaMalloc((void **) &(d_pnodored), tam_nodo * sizeof(struct NodoRed));
    cudaMalloc((void **) &(d_pidnodo), tam_nodo * sizeof(long));
    cudaMalloc((void **) &(d_psubnodo), tam_nodo * sizeof(struct SubNodo));
    cudaMalloc((void **) &(d_presrec), tam_nodo * sizeof(struct ResRec));


    cudaMemcpy(d_pnodored, h_pnodored, tam_nodo * sizeof(struct NodoRed), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pidnodo, h_pidnodo, tam_nodo * sizeof(long), cudaMemcpyHostToDevice);

    std::cout << "Vuelve indices a id_nodos ..." << std::endl;


    int ii = 1;
    unsigned int delta = 200000;
    unsigned int tam_n1;

    for (unsigned int tam_n0 = 0; tam_n0 < tam_nodo; tam_n0 += delta) {
        tam_n1 = (tam_n0 + delta > tam_nodo) ? tam_nodo : tam_n0 + delta;
        std::cout << "ciclo: " << ii << " " << tam_n0 << " " << tam_n1 << std::endl;
        ii++;
        kernel_indexaNodos<<<canti_bloques, canti_hilos>>>(d_pnodored, d_psubnodo, d_pidnodo, tam_nodo, tam_n0, tam_n1);
        cudaDeviceSynchronize();
    }


    std::cout << "Prepara el destino para registrar las distancias ..." << std::endl;

    kernel_preparaDist<<<canti_bloques, canti_hilos>>>(d_pnodored, d_presrec, tam_nodo);
    cudaDeviceSynchronize();


    std::cout << "Cálcula distancias a nodos ..." << std::endl;

    double variacion1, variacion0;
    int numiters = 600;
    do {
        variacion0 = variacion1;
        kernel_principalDN<<<canti_bloques, canti_hilos>>>(d_pnodored, d_psubnodo, d_presrec, tam_nodo);
        variacion1 = sumaVariacion(d_presrec, tam_nodo);
        std::cout << "Ciclo cálculo:" << numiters << " " << variacion0 - variacion1 << std::endl;
        numiters--;
    } while (variacion0 != variacion1 && numiters > 0);


    cudaMemcpy(h_presrec, d_presrec, tam_nodo * sizeof(struct ResRec), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_psubnodo, d_psubnodo, tam_nodo * sizeof(struct SubNodo), cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < tam_nodo; i++) {
        (pnodo + i)->id_rec = (h_presrec + i)->id_rec;
        (pnodo + i)->dist_rec = (h_presrec + i)->dist_rec;

        for (int j = 0; j < (h_psubnodo + i)->nsn; j++) {
            (pnodo + i)->aindex_nodos[j] = (h_psubnodo + i)->aindex_nodos[j];
        }
    }


    cudaFree(d_pnodored);
    cudaFree(d_psubnodo);
    cudaFree(d_pidnodo);
    cudaFree(d_presrec);

    free(h_pnodored);
    free(h_pidnodo);
    free(h_psubnodo);
    free(h_presrec);
}


