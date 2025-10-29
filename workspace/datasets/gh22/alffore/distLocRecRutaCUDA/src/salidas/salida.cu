//
// Created by alfonso on 26/09/21.
//

#include "../distlocRuta.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstdio>
#include <cstdlib>

//const static double RT = 6371000.00;

/**
 *
 * @param pNodo
 * @param pLoc
 * @param tam_loc
 * @param snomarch
 */
void salidaLoc(PNodo pNodo, PLocalidad pLoc, unsigned int tam_loc, std::string snomarch) {

    std::ofstream fssal;

    fssal.open(snomarch.c_str());

    for (unsigned int i = 0; i < tam_loc; i++) {
        PLocalidad p = pLoc + i;

        PNodo pn = pNodo + p->index_nodo;

        fssal.unsetf(std::ios::fixed | std::ios::scientific);
        fssal << p->id_loc << "," << pn->id_rec << ",";
        //fssal << std::ios::fixed << std::ios::showpoint << std::setprecision(4) << pn->dist_rec << "," << p->dist_nodo              << std::endl;
        fssal << std::setprecision(4) << pn->dist_rec << "," << p->dist_nodo << std::endl;
    }

    fssal.close();
}

void salidaLoc_v2(PNodo pNodo, PLocalidad pLoc, unsigned int tam_loc, std::string snomarch) {

    FILE *fp;
    fp=fopen(snomarch.c_str(),"w");

    for (unsigned int i = 0; i < tam_loc; i++) {
        PLocalidad p = pLoc + i;

        PNodo pn = pNodo + p->index_nodo;
        fprintf(fp,"%li,%li,%.4lf\n",p->id_loc,pn->id_rec,pn->dist_rec);
    }
    fclose(fp);

}

void salidaNodos(PNodo pNodo,unsigned int tam_nodo,std::string snomarch){
    FILE *fp;

    fp = fopen(snomarch.c_str(),"w");

    for(unsigned int i=0;i<tam_nodo;i++){
        PNodo p =pNodo+i;
        fprintf(fp,"%li,%li,%lf\n",p->id_nodo,p->id_rec,p->dist_rec);
    }

    fclose(fp);
}