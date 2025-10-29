//
// Created by alfonso on 12/10/21.
//

#include "../distlocRutaCUDA.h"

/**
 *
 * @param d_pseg
 * @param d_preln
 * @param d_pidnodo
 * @param tam_nodo
 * @param tam_seg
 */
__global__ void
kernel_generaRed(PSegmento d_pseg, PDRelNodo d_preln,const long *d_pidnodo, unsigned int tam_seg, unsigned int tam_nodo) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= tam_nodo)return;

    int n = 0;

    for (unsigned int iseg = 0; iseg < tam_seg; iseg++) {

        if ((d_pseg + iseg)->id_nodo1 == *(d_pidnodo + idx)) {
            (d_preln + idx)->aid_nodos[n] = (d_pseg + iseg)->id_nodo0;
            (d_preln + idx)->adist[n] = (d_pseg + iseg)->dist;
            n++;
        } else if ((d_pseg + iseg)->id_nodo0 == *(d_pidnodo + idx)) {
            (d_preln + idx)->aid_nodos[n] = (d_pseg + iseg)->id_nodo1;
            (d_preln + idx)->adist[n] = (d_pseg + iseg)->dist;
            n++;
        }

    }

    (d_preln + idx)->nsn = n;
}


/**
 *
 * @param h_pseg
 * @param pnodo
 * @param tam_seg
 * @param tam_nodo
 */
void generaRedNodos(PSegmento h_pseg, PNodo pnodo, unsigned int tam_seg, unsigned int tam_nodo) {

    std::cout << "Relaciona Nodos para formar Red ..." << std::endl;

    int canti_hilos = 1000;
    int canti_bloques = (int) ceil((double) tam_nodo / canti_hilos);


    std::cout << "Hilos: " << canti_hilos << " Bloques: " << canti_bloques << std::endl;

    long *d_pidnodo;
    PDRelNodo d_preln;
    PSegmento d_pseg;

    auto h_pidnodo = (long *) malloc(sizeof(long) * tam_nodo);
    auto h_preln = (PDRelNodo) malloc(sizeof(struct DRelNodo) * tam_nodo);

    for (size_t i = 0; i < tam_nodo; i++) {
        *(h_pidnodo + i) = (pnodo + i)->id_nodo;
    }

    cudaMalloc((void **) &(d_pseg), tam_seg * sizeof(struct Segmento));
    cudaMalloc((void **) &(d_preln), tam_nodo * sizeof(struct DRelNodo));
    cudaMalloc((void **) &(d_pidnodo), tam_nodo * sizeof(long));

    cudaMemcpy(d_pidnodo, h_pidnodo, tam_nodo * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pseg, h_pseg, tam_seg * sizeof(struct Segmento), cudaMemcpyHostToDevice);

    kernel_generaRed<<<canti_bloques, canti_hilos>>>(d_pseg, d_preln,(const long*) d_pidnodo, tam_seg, tam_nodo);

    cudaMemcpy(h_preln, d_preln, tam_nodo * sizeof(struct DRelNodo), cudaMemcpyDeviceToHost);

    for (auto i = 0; i < tam_nodo; i++) {
        PDRelNodo prel = h_preln + i;
        if (prel->nsn > 0) {
            PNodo pn = pnodo + i;
            pn->nsn = prel->nsn;
            for (int j = 0; j < pn->nsn; j++) {
                pn->adist[j] = prel->adist[j];
                pn->aid_nodos[j] = prel->aid_nodos[j];
            }
        }
    }

    cudaFree(d_pidnodo);
    cudaFree(d_preln);
    cudaFree(d_pseg);

    free(h_preln);
    free(h_pidnodo);
}