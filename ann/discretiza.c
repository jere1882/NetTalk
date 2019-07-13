/*
PMG - Ultima revision: 01/06/2001
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

/*Defino la funcion sigmoidea*/
#define sigmoid(h) 1.0/(1.0+exp(-h))

/*parametros de la red y el entrenamiento*/
int N1;             /* N1: NEURONAS EN CAPA DE ENTRADA */
int N2;             /* N2: NEURONAS EN CAPA INTERMEDIA */
int N3;             /* N2: NEURONAS EN CAPA DE SALIDA  */

int ITER;           /* Total de Iteraciones*/
float ETA;          /* learning rate */
float u;            /* Momentum */
int NERROR;         /* graba error cada NERROR iteraciones */
int WTS;            /* numero de archivo de sinapsis inicial
                       WTS=0 implica empezar la red con valores de sinapsis al azar*/

int PTOT;           /* cantidad TOTAL de patrones en el archivo .data */
int PR;             /* cantidad de patrones de ENTRENAMIENTO */
int PTEST;          /* cantidad de patrones de TEST (archivo .test) */
                    /* cantidad de patrones de VALIDACION: PTOT - PR*/

int SEED;           /* semilla para la funcion rand(). Los posibles valores son:*/
                    /* SEED: -1: No mezclar los patrones: usar los primeros PR para entrenar
                                 y el resto para validar.Toma la semilla del rand con el reloj.
                              0: Seleccionar semilla con el reloj, y mezclar los patrones.
                             >0: Usa el numero leido como semilla, y mezcla los patrones. */

int CONTROL;        /* nivel de verbosity: 0 -> solo resumen, 1 -> 0 + pesos, 2 -> 1 + datos*/


/*matrices globales*/
float **test;                     /* test  data */

/*variables globales auxiliares*/
char filepat[100];
int i,j,k;
float h;
/*bandera de error*/
int error;


/* -------------------------------------------------------------------------- */
/*define_matrix: reserva espacio en memoria para todas las matrices declaradas.
  Todas las dimensiones son leidas del archivo .net en la funcion arquitec()  */
/* -------------------------------------------------------------------------- */
int define_matrix(){

  if(PTEST) test=(float **)calloc(PTEST,sizeof(float *));
  if((PTEST&&test==NULL)) return 1;

  for(i=0;i<PTEST;i++){
    test[i]=(float *)calloc(N1+N3+1,sizeof(float *));
    if(test[i]==NULL) return 1;
  }

  return 0;
}
/* ---------------------------------------------------------------------------------- */
/*arquitec: Lee el archivo .net e inicializa la red en funcion de los valores leidos
  filename es el nombre del archivo .net (sin la extension) */
/* ---------------------------------------------------------------------------------- */
int arquitec(char *filename){
  FILE *b;

  /*Paso 1:leer el archivo con la configuracion*/
  sprintf(filepat,"%s.net",filename);
  b=fopen(filepat,"r");
  error=(b==NULL);
  if(error){
    printf("Error al abrir el archivo de parametros\n");
    return 1;
  }

  /* Estructura de la red */
  fscanf(b,"%d",&N1);
  fscanf(b,"%d",&N2);
  fscanf(b,"%d",&N3);

  /* Archivo de patrones: datos para train y para validacion */
  fscanf(b,"%d",&PTOT);
  fscanf(b,"%d",&PR);
  fscanf(b,"%d",&PTEST);

  /* Parametros para el entrenamiento */
  fscanf(b,"%d",&ITER);
  fscanf(b,"%f",&ETA);
  fscanf(b,"%f",&u);

  /* Datos para la salida de resultados */
  fscanf(b,"%d",&NERROR);

  /* Inicializacion de las sinapsis - Azar o Archivo */
  fscanf(b,"%d",&WTS);

  /* Semilla para la funcion rand()*/
  fscanf(b,"%d",&SEED);

  /* Nivel de verbosity*/
  fscanf(b,"%d",&CONTROL);

  fclose(b);


  /*Paso 2: Definir matrices para datos y pesos*/
  error=define_matrix();
  if(error){
    printf("Error en la definicion de matrices\n");
    return 1;
  }


  /*Imprimir control por pantalla*/
  printf("\nArquitectura de la red: %d:%d:%d",N1,N2,N3);
  printf("\nArchivo de patrones: %s",filename);
  printf("\nCantidad total de patrones: %d",PTOT);
  printf("\nCantidad de patrones de entrenamiento: %d",PR);
  printf("\nCantidad de patrones de validacion: %d",PTOT-PR);
  printf("\nCantidad de patrones de test: %d",PTEST);

  return 0;
}
/* -------------------------------------------------------------------------------------- */
/*convierte_data: lee los datos de el archivo de prediccion(.predic)
  y lo grava en un nuevo archivo .predic.d con la salida discretizada (0 o 1)
  filename es el nombre de los archivos (sin extension)
  La cantidad de datos y la estructura de los archivos fue leida en la funcion arquitec()
  Los registros en el archivo pueden estar separados por blancos ( o tab ) o por comas    */
/* -------------------------------------------------------------------------------------- */
int convierte_data(char *filename){

  FILE *fpred;
  FILE *salida;
  float valor;
  int separador;

  sprintf(filepat,"%s.predic",filename);
  fpred=fopen(filepat,"r");
  error=(fpred==NULL);
  if(error){
    printf("Error al abrir el archivo de predicciones\n");
    return 1;
  }
  sprintf(filepat,"%s.predic.d",filename);
  salida=fopen(filepat,"w");
  error=(salida==NULL);
  if(error){
    printf("Error al abrir el archivo de salida\n");
    return 1;
  }

  if(CONTROL>1) printf("\n\nDatos de prediccion:");

  for(k=0;k<PTEST;k++){
    /*leer el patron*/
    if(CONTROL>1) printf("\nP%d:\t",k);
    test[k][0]=-1.0;
    for(i=1;i<=N1+N3;i++){
      fscanf(fpred,"%f",&valor);
      test[k][i]=valor;
      if(CONTROL>1) printf("%f\t",test[k][i]);
      separador=getc(fpred);
      if(separador!=',') ungetc(separador,fpred);
    }

    /*grabar el patron en el archivo de salida*/
    for(i=1;i<=N1;i++) fprintf(salida,"%f\t",test[k][i]);
    fprintf(salida,"%d\n",test[k][i]<0.5 ? 0 : 1 );
    if(CONTROL>1) printf("->%d",test[k][i]<0.5 ? 0 : 1 );

  }
  fclose(fpred);
  fclose(salida);
  return 0;

}



/* ----------------------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
int main(int argc, char **argv){

  if(argc!=2){
    printf("Modo de uso: separa_clases <filename>\ndonde filename es el nombre del archivo (sin extension)\n");
    return 0;
  }

  printf("Discretizar clases\n");

  /* defino la red e inicializo los pesos */
  error=arquitec(argv[1]);
  if(error){
    printf("Error en la definicion de la red\n");
    return 1;
  }

  if(!PTEST){
    printf("No hay puntos en prediccion: PTEST=0!!\n");
    return 1;
  }

  /* convierto los datos */
  error=convierte_data(argv[1]);                 
  if(error){
    printf("Error en el proceso de datos\n");
    return 1;
  }

  printf("\nDiscretizar clases terminado\n");
 
  return 0;

}
/* ----------------------------------------------------------------------------------------------------- */












