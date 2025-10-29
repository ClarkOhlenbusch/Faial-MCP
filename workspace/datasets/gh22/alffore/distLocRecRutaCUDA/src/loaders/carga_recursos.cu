//
// Created by alfonso on 14/09/21.
//

#include "../distlocRuta.h"

int cargaArchivoRec(std::string snomarchivo, PRecurso prec);

void parseaLineaRec(char *slinea, PRecurso prec);

void convierteCoordRec(PRecurso p, double lon, double lat);

/**
 *
 * @param snomarchivo
 * @param prec
 * @return
 */
int cargaArchivoRec(std::string snomarchivo, PRecurso prec) {

    printf("Carga Recursos ...\n");

    FILE *fp = fopen(snomarchivo.c_str(), "r");
    unsigned int pos = 0;

    if (fp != NULL) {
        char buffer[MAX_LONG];

        while (fgets(buffer, MAX_LONG, fp)) {
            parseaLineaRec(buffer, prec + pos);
            pos++;
        }
    }

    fclose(fp);
    return 0;
}

/**
 *
 * @param slinea
 * @param prec
 */
void parseaLineaRec(char *slinea, PRecurso prec){
    char *result = NULL;
    double lon,lat;
    result = strtok(slinea, SEP);

    lon = atof(result);
    result = strtok(NULL, SEP);

    lat = atof(result);
    result = strtok(NULL, SEP);

    prec->id_rec = atoi(result);

    convierteCoordRec(prec,lon,lat);

    prec->dist_nodo = MAX_DIST;
    prec->id_nodo = -1;
    prec->index_nodo=-1;
    
}

/**
 *
 * @param p
 * @param lon
 * @param lat
 */
void convierteCoordRec(PRecurso p, double lon, double lat){
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