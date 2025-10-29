
#include "../distlocRuta.h"


PLocalidad pLoc;
PRecurso pRec;
PNodo pNodo;
PSegmento pSeg;
PResLoc pResLoc;


const size_t tam_loc = 292447;
const size_t tam_nodo = 1560802;
const size_t tam_rec = 1407;
const size_t tam_seg = 1756216;


//funciones externas
extern int cargaArchivoLoc(std::string snomarchivo, PLocalidad ploc);

extern int cargaArchivoRec(std::string snomarchivo, PRecurso prec);

extern int cargaArchivoNodo(std::string snomarchivo, PNodo pnodo);

extern int cargaArchivoSeg(std::string snomarchivo, PSegmento pseg);

extern void relacionaLoc2Nodo(PLocalidad ploc, PNodo pnodo, unsigned int tam_loc, unsigned int tam_nodo);

extern void relacionaRec2Nodo(PRecurso prec, PNodo pnodo, unsigned int tam_rec, unsigned int tam_nodo);

extern void generaRedNodos(PSegmento h_pseg, PNodo pnodo, unsigned int tam_seg, unsigned int tam_nodo);

extern void generaRutas(PNodo pnodo, unsigned int tam_nodo);

extern void salidaLoc(PNodo pNodo, PLocalidad pLoc,unsigned int tam_loc, std::string snomarch);

extern void salidaLoc_v2(PNodo pNodo, PLocalidad pLoc, unsigned int tam_loc, std::string snomarch);
extern void salidaNodos(PNodo pNodo,unsigned int tam_nodo,std::string snomarch);

/**
 *
 */
void alojaHostMemoria() {
    pLoc = (PLocalidad) malloc(sizeof(struct Localidad) * tam_loc);
    pNodo = (PNodo) malloc(sizeof(struct Nodo) * tam_nodo);
    pRec = (PRecurso) malloc(sizeof(struct Recurso) * tam_rec);
    pSeg = (PSegmento) malloc(sizeof(struct Segmento) * tam_seg);
    pResLoc = (PResLoc) malloc(sizeof(struct ResLoc) * tam_nodo);
}

/**
 *
 */
void liberaHostMemoria() {
    free(pLoc);
    free(pNodo);
    free(pRec);
    free(pSeg);
    free(pResLoc);
}

void cargaRecursos() {
    std::string sarchivo_loc = "/home/alfonso/devel/carreteras/conjunto_de_datos/localidad.csv";
    cargaArchivoLoc(sarchivo_loc, pLoc);

    std::string sarchivo_nodo = "/home/alfonso/devel/carreteras/conjunto_de_datos/union_min.csv";
    cargaArchivoNodo(sarchivo_nodo, pNodo);

    std::string sarchivo_rec = "/home/alfonso/devel/renic/renic.git/utiles/cac_ruta/museo.csv";
    cargaArchivoRec(sarchivo_rec, pRec);

    std::string sarchivo_seg = "/home/alfonso/devel/lectorINTCarr/rutas.csv";
    cargaArchivoSeg(sarchivo_seg, pSeg);
}

/**
 *
 */
void calculos() {

    relacionaLoc2Nodo(pLoc, pNodo, tam_loc, tam_nodo);

    relacionaRec2Nodo(pRec, pNodo, tam_rec, tam_nodo);

    generaRedNodos(pSeg, pNodo, tam_seg, tam_nodo);

    generaRutas(pNodo, tam_nodo);
    for (size_t i = 0; i < tam_nodo; i++) {
        PNodo p = pNodo + i;
        std::cout << "Nodo Localizado: " << i << " id_rec:" << p->id_rec << " id_nodo:" << p->id_nodo << " dist_rec:"
                  << p->dist_rec << " " << std::endl;
    }
}

/**
 *
 */
void generaSalidas(){
    salidaLoc(pNodo,pLoc,tam_loc,"salida_loc.csv");
    salidaLoc_v2(pNodo,pLoc,tam_loc,"salida_loc_v2.csv");
    salidaNodos(pNodo,tam_nodo,"salida_nodos_cuda.csv");
}

/**
 *
 * @return
 */
int main() {

    alojaHostMemoria();

    cargaRecursos();

    calculos();

    generaSalidas();

    liberaHostMemoria();

    return 0;
}