//
// Created by alfonso on 16/09/21.
//
#include "../distlocRuta.h"

int cargaArchivoSeg(std::string snomarchivo, PSegmento pseg);

void parseaLineaSeg(char *slinea, PSegmento pseg);


/**
 *
 * @param snomarchivo
 * @param pseg
 * @return
 */
int cargaArchivoSeg(std::string snomarchivo, PSegmento pseg) {

    printf("Carga Segmentos ...\n");

    FILE *fp = fopen(snomarchivo.c_str(), "r");
    unsigned int pos = 0;

    if (fp != NULL) {
        char buffer[MAX_LONG];

        while (fgets(buffer, MAX_LONG, fp)) {
            parseaLineaSeg(buffer, pseg + pos);
            pos++;
            //std::cout<<pos<<": "<<buffer<<std::endl;
        }
    }

    fclose(fp);
    return 0;
}

/**
 *
 * @param slinea
 * @param pseg
 */
void parseaLineaSeg(char *slinea, PSegmento pseg){
    char *result = NULL;

    strtok(slinea, SEP);

    result = strtok(NULL, SEP);

    double lon0 = atof(result);
    result = strtok(NULL, SEP);

    double lat0 = atof(result);
    result = strtok(NULL, SEP);

    double lon1 = atof(result);
    result = strtok(NULL, SEP);

    double lat1 = atof(result);
    result = strtok(NULL, SEP);

    pseg->id_nodo0 = atoi(result);
    result = strtok(NULL, SEP);

    pseg->id_nodo1 = atoi(result);
    result = strtok(NULL, SEP);

    pseg->dist = atof(result);

}