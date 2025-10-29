//
// Created by alfonso on 25/09/21.
//

#include "../distlocRutaCUDA.h"

/**
 * Función que marca subnodos con distancia
 * @param d_pnodored
 * @param d_psubnodo
 * @param d_presrec
 * @param idx
 * @param dist
 * @param irec
 */
__device__ void
recorrido(PNodoRed d_pnodored, PSubNodo d_psubnodo, PResRec d_presrec, unsigned int idx, double dist, long irec) {

    if ((d_psubnodo + idx)->nsn == 0)return;

    for (int i = 0; i < (d_psubnodo + idx)->nsn; i++) {

        unsigned int index = (d_psubnodo + idx)->aindex_nodos[i];
        if (index != idx) {
            double nueva_dist = (d_pnodored + idx)->adist[i] + dist;

            if ((d_presrec + index)->id_rec < 0 ||
                ((d_presrec + index)->id_rec > 0 && (d_presrec + index)->dist_rec > nueva_dist)) {

                (d_presrec + index)->delta += nueva_dist - (d_presrec + index)->dist_rec;
                (d_presrec + index)->dist_rec = nueva_dist;
                (d_presrec + index)->id_rec = irec;

            }
        }
    }
}


/**
 * Kernel principal para marcar y propagar las distancias desde los nodos que poseen un recurso
 * @param d_pnodored
 * @param d_psubnodo
 * @param d_presrec
 * @param tam_nodo
 */
__global__ void kernel_principalDN(PNodoRed d_pnodored, PSubNodo d_psubnodo, PResRec d_presrec, unsigned int tam_nodo) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= tam_nodo)return;

    if ((d_presrec + idx)->id_rec > 0) {
        recorrido(d_pnodored, d_psubnodo, d_presrec, idx, (d_presrec + idx)->dist_rec, (d_presrec + idx)->id_rec);
    }
}
