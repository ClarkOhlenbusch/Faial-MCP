//
// Created by alfonso on 16/09/21.
//
#include "../distlocRutaCUDA.h"

const static double RT = 6371000.00;

void relacionaRec2Nodo(PRecurso prec, PNodo pnodo, unsigned int tam_rec, unsigned int tam_nodo);

/**
 *
 * @param d_pnodo
 * @param d_prec
 * @param d_presu
 * @param tam_loc
 * @param tam_n0
 * @param tam_n1
 */
__global__ void
calculadistR2Nv2(PDNodo d_pnodo, PDRecurso d_prec, PDResu d_presu, unsigned int tam_loc, unsigned int tam_n0,
                 unsigned int tam_n1) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double dmin = 8;
    long jmin = -1;

    if (idx >= tam_loc)return;
    PDRecurso p = d_prec + idx;

    if (tam_n0 > 0) {
        dmin = (d_presu + idx)->dist;
        jmin = (d_presu + idx)->index;
    }

    for (unsigned int j = tam_n0; j < tam_n1; j++) {
        PDNodo pn = d_pnodo + j;
        double daux = pn->x * p->x + pn->y * p->y + pn->z * p->z;
        if (daux > 1.00)daux = 1.00;
        if (daux < -1.00)daux = -1.00;
        daux = acos(daux);
        if (dmin > daux) {
            dmin = daux;
            jmin = j;
        }
    }

    (d_presu + idx)->dist = RT * dmin;
    (d_presu + idx)->index = jmin;
}

/**
 *
 * @param prec
 * @param pnodo
 * @param tam_rec
 * @param tam_nodo
 */
void relacionaRec2Nodo(PRecurso prec, PNodo pnodo, unsigned int tam_rec, unsigned int tam_nodo) {

    std::cout << "Relaciona Recursos con Nodos ..." << std::endl;

    int canti_hilos = 840;
    int canti_bloques = (int) ceil((double) tam_rec / canti_hilos) + 1;

    unsigned int delta = 100000;
    unsigned int tam_n1;

    std::cout << "Hilos: " << canti_hilos << " Bloques: " << canti_bloques << std::endl;

    PDNodo d_pnodo = nullptr;
    PDRecurso d_prec = nullptr;
    PDResu d_presu = nullptr;


    auto h_pnodo = (PDNodo) malloc(sizeof(struct DNodo) * tam_nodo);
    auto h_prec = (PDRecurso) malloc(sizeof(struct DRecurso) * tam_rec);
    auto h_presu = (PDResu) malloc(sizeof(struct DResu) * tam_rec);

    for (auto i = 0; i < tam_rec; i++) {
        (h_prec + i)->x = (prec + i)->x;
        (h_prec + i)->y = (prec + i)->y;
        (h_prec + i)->z = (prec + i)->z;
    }

    for (auto i = 0; i < tam_nodo; i++) {
        (h_pnodo + i)->x = (pnodo + i)->x;
        (h_pnodo + i)->y = (pnodo + i)->y;
        (h_pnodo + i)->z = (pnodo + i)->z;
    }

    cudaMalloc((void **) &(d_prec), tam_rec * sizeof(struct DRecurso));
    cudaMalloc((void **) &(d_pnodo), tam_nodo * sizeof(struct DNodo));
    cudaMalloc((void **) &(d_presu), tam_rec * sizeof(struct DResu));

    cudaMemcpy(d_prec, h_prec, tam_rec * sizeof(struct DRecurso), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pnodo, h_pnodo, tam_nodo * sizeof(struct DNodo), cudaMemcpyHostToDevice);

    int ii = 1;
    for (unsigned int tam_n0 = 0; tam_n0 < tam_nodo; tam_n0 += delta) {
        tam_n1 = (tam_n0 + delta > tam_nodo) ? tam_nodo : tam_n0 + delta;
        std::cout << "ciclo: " << ii << " " << tam_n0 << " " << tam_n1 << std::endl;
        ii++;
        calculadistR2Nv2<<<canti_bloques, canti_hilos>>>(d_pnodo, d_prec, d_presu, tam_rec, tam_n0, tam_n1);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_presu, d_presu, tam_rec * sizeof(struct DResu), cudaMemcpyDeviceToHost);

    for (auto i = 0; i < tam_rec; i++) {
        (prec + i)->dist_nodo = (h_presu + i)->dist;
        (prec + i)->index_nodo = (h_presu + i)->index;
        (prec + i)->id_nodo = (pnodo + (prec + i)->index_nodo)->id_nodo;
        (pnodo + (prec + i)->index_nodo)->id_rec = (prec + i)->id_rec;
        (pnodo + (prec + i)->index_nodo)->dist_rec = (prec + i)->dist_nodo;
    }

    cudaFree(d_presu);
    cudaFree(d_pnodo);
    cudaFree(d_prec);

    free(h_presu);
    free(h_prec);
    free(h_pnodo);
}
