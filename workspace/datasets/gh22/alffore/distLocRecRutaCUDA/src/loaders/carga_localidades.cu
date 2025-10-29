//
// Created by alfonso on 14/09/21.
//

#include "../distlocRuta.h"

int cargaArchivoLoc(std::string snomarchivo, PLocalidad ploc);

void parseaLineaLoc(char *slinea, PLocalidad ploc);

void convierteCoordLoc(PLocalidad p, double lon, double lat);


/**
 * Función que carga el archivo de Localidades
 */
int cargaArchivoLoc(std::string snomarchivo, PLocalidad ploc) {

    printf("Carga localidades ...\n");

    FILE *fp = fopen(snomarchivo.c_str(), "r");
    unsigned int pos = 0;

    if (fp != NULL) {
        char buffer[MAX_LONG];

        while (fgets(buffer, MAX_LONG, fp)) {
            parseaLineaLoc(buffer, ploc + pos);
            pos++;
        }
    }

    fclose(fp);
    return 0;
}

/**
 *
 */
void parseaLineaLoc(char *slinea, PLocalidad ploc) {
    char *result = NULL;
    double lon,lat;

    result = strtok(slinea, SEP);

    lon = atof(result);
    result = strtok(NULL, SEP);

    lat = atof(result);
    result = strtok(NULL, SEP);

    result = strtok(NULL, SEP);

    ploc->id_loc = atoi(result);

    convierteCoordLoc(ploc,lon,lat);


    ploc->id_nodo = -1;
    ploc->dist_nodo = MAX_DIST;
    ploc->index_nodo=-1;
}


/**
 *
 */
void convierteCoordLoc(PLocalidad p, double lon, double lat) {

    double r;

    lat = lat * M_PI / 180.0;
    lon = lon * M_PI / 180.0;

    p->x = sin(lon) * cos(lat);
    p->y = cos(lon) * cos(lat);
    p->z = sin(lat);

    r = pow(p->x, 2.0) + pow(p->y, 2.0) + pow(p->z, 2.0);
    r = sqrt(r);

    p->x /= r;
    p->y /= r;
    p->z /= r;
}

