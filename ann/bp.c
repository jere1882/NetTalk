/*
BP: Entrenamiento de redes neuronales feedforward de 3 capas (entrada - oculta - salida).
Algoritmo: Backpropagation estocastico.
Capa intermedia con unidades sigmoideas.
Salidas con unidades lineales.

PMG - Ultima revision: 18/02/2002
*/

#include <stdio.h>
#include <string.h>
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
int NERROR;         /* guarda mediciones de error cada NERROR iteraciones */
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

int ARTICULATORY_FEATURES=18;
int PHONEMS=51;


float discrete_error;


/*matrices globales*/
float **data;                 /* train data */
float **test;                 /* test  data */
float **w1,**w2;	      	  /* pesos entre neuronas */
float *grad2,*grad3;	      /* gradiente en cada unidad */
float **dw1,**dw2;	          /* correccion a cada peso   */
float *x1,*x2,*x3;    	      /* activaciones de cada capa   */
float **pred;                 /* salidas predichas */
float *target; 	       	      /* valor correcto de la salida */
float **afs;                    /* phonems to articulatory features */
int *seq;      	       		  /* sequencia de presentacion de los patrones*/

/*variables globales auxiliares*/
char filepat[100];
/*bandera de error*/
int error;

/* ------------------------------------------------------------------------------- */
/*unif_rnd:         
  genero un numero al azar en [-max , max]                                         */
/* ------------------------------------------------------------------------------- */

float unif_rnd(float max){
    return  (rand()/(RAND_MAX/2.0) - 1.0) * max;
}

/* ------------------------------------------------------------------------------- */
/*sinapsis_rnd: Inicializa al azar los valores de todas las sinapsis                 
  Los valores se toman entre [-max , max]
  SEED contiene la semilla del rand(). Si SEED es <=0 se toma como semilla el reloj*/
/* ------------------------------------------------------------------------------- */
void sinapsis_rnd(float max){
  int i,j;
  float x;
  time_t t; 

  /* Semilla para la funcion rand() */
  if(SEED<0) srand((unsigned) time(&t));   
  else{
    if(SEED==0) SEED=time(&t);
    srand((unsigned)SEED);
  }  

  /*primer capa*/
  for (j = 1; j <= N2; j ++) for (i = 0; i <= N1; i ++) w1[j][i] = unif_rnd(max);


  /*segunda capa*/
  for (j = 1; j <= N3; j ++) for (i = 0; i <= N2; i ++) w2[j][i] = unif_rnd(max);

}
/* ------------------------------------------------------------------- */
/*sinapsis_save: guarda valores de las sinapsis en un archivo.
  El nombre del archivo de salida es (WTS).wts, donde WTS es un entero.*/
/* ------------------------------------------------------------------- */
int sinapsis_save(int WTS)  {
  FILE *fp;
  int i,j,largo;
  char p[13];
  
  sprintf(p, "%d", WTS);
  largo=strlen(p);
  p[largo]='.';
  p[largo+1]='w';
  p[largo+2]='t';
  p[largo+3]='s';
  p[largo+4]='\0';
  
  if((fp=fopen(p,"w"))==NULL) return 1;
  
  /*primer capa*/
  for(j=1;j<=N2;j++)  for(i=0;i<=N1;i++) fprintf(fp,"%f\n",w1[j][i]);
  
  /*segunda capa*/
  for (i=1;i<=N3;i++) for(j=0;j<=N2;j++) fprintf(fp,"%f\n",w2[i][j]);
  
  fclose(fp);

  return 0;
}
/* --------------------------------------------------------------------- */
/*sinapsis_read: lee valores de las sinapsis desde un archivo.
  El nombre del archivo de entrada es (WTS).wts , donde WTS es un entero.*/
/* --------------------------------------------------------------------- */
int sinapsis_read(int WTS)  {
  FILE *fp;
  int i,j,largo;
  char p[13];

  sprintf(p, "%d", WTS);
  largo=strlen(p);
  p[largo]='.';
  p[largo+1]='w';
  p[largo+2]='t';
  p[largo+3]='s';
  p[largo+4]='\0';
  if((fp=fopen(p,"r"))==NULL) return 1;
  
  /*primera capa*/
  for(j=1;j<=N2;j++) for(i=0;i<=N1;i++) fscanf(fp,"%f",&w1[j][i]);
  
  /*segunda capa*/
  for (i=1;i<=N3;i++) for(j=0;j<=N2;j++) fscanf(fp,"%f",&w2[i][j]);
  
  fclose(fp);

  return 0;
}
/* -------------------------------------------------------------------------- */
/*define_matrix: reserva espacio en memoria para todas las matrices declaradas.
  Todas las dimensiones son leidas del archivo .net en la funcion arquitec()  */
/* -------------------------------------------------------------------------- */
int define_matrix(){

  int i,j,max;
  
  if(PTOT>PTEST) max=PTOT;
  else max=PTEST;

  seq=(int *)calloc(max,sizeof(int));
  x1=(float *)calloc(N1+1,sizeof(float));
  x2=(float *)calloc(N2+1,sizeof(float));
  x3=(float *)calloc(N3+1,sizeof(float));
  grad2=(float *)calloc(N2+1,sizeof(float));
  grad3=(float *)calloc(N3+1,sizeof(float));
  target=(float *)calloc(N3+1,sizeof(float));
  if(seq==NULL||x1==NULL||x2==NULL||x3==NULL||grad2==NULL||grad3==NULL||target==NULL) return 1;
  
  w1= (float **)calloc(N2+1,sizeof(float *));
  w2= (float **)calloc(N3+1,sizeof(float *));
  dw1= (float **)calloc(N2+1,sizeof(float *));
  dw2= (float **)calloc(N3+1,sizeof(float *));
  data=(float **)calloc(PTOT,sizeof(float *));
  if(PTEST) test=(float **)calloc(PTEST,sizeof(float *));
  pred=(float **)calloc(max,sizeof(float *));
  afs = (float **)calloc(PHONEMS,sizeof(float*));
  if(w1==NULL||w2==NULL||dw1==NULL||dw2==NULL||data==NULL||(PTEST&&test==NULL)||pred==NULL) return 1;

  for(i=0;i<=N2;i++){
    w1[i]=(float *)calloc(N1+1,sizeof(float));
    dw1[i]=(float *)calloc(N1+1,sizeof(float));
	if(w1[i]==NULL||dw1[i]==NULL) return 1;
  }
  for(i=0;i<=N3;i++){
    w2[i]=(float *)calloc(N2+1,sizeof(float));
    dw2[i]=(float *)calloc(N2+1,sizeof(float));
	if(w2[i]==NULL||dw2[i]==NULL) return 1;
  }

  for(i=0;i<PTOT;i++){
    data[i]=(float *)calloc(N1+N3+1,sizeof(float));
	if(data[i]==NULL) return 1;
  }
  for(i=0;i<PTEST;i++){
    test[i]=(float *)calloc(N1+N3+1,sizeof(float));
	if(test[i]==NULL) return 1;
  }
  for(i=0;i<max;i++){
    pred[i]=(float *)calloc(N3+1,sizeof(float));
	if(pred[i]==NULL) return 1;
  }
  for(i=0;i<PHONEMS;i++){
  	afs[i] =(float *)calloc(ARTICULATORY_FEATURES,sizeof(float));
  }
  return 0;
}
/* ---------------------------------------------------------------------------------- */
/*arquitec: Lee el archivo .net e inicializa la red en funcion de los valores leidos
  filename es el nombre del archivo .net (sin la extension) */
/* ---------------------------------------------------------------------------------- */
int arquitec(char *filename){
  FILE *b;
  int i,j;

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

  /*Paso 3:leer sinapsis desde archivo o iniciar al azar*/
  if(WTS!=0) error=sinapsis_read(WTS);
  else sinapsis_rnd(0.1);
  if(error){
    printf("Error en la lectura de los pesos desde archivo\n");
    return 1;
  }

  /*Imprimir control por pantalla*/
  printf("\nArquitectura de la red: %d:%d:%d",N1,N2,N3);
  printf("\nArchivo de patrones: %s",filename);
  printf("\nCantidad total de patrones: %d",PTOT);
  printf("\nCantidad de patrones de entrenamiento: %d",PR);
  printf("\nCantidad de patrones de validacion: %d",PTOT-PR);
  printf("\nCantidad de patrones de test: %d",PTEST);
  printf("\nEpocas: %d",ITER);
  printf("\nLearning Rate: %f",ETA);
  printf("\nMomentum: %f",u);
  printf("\nFrecuencia para grabar resultados: %d EPOCAS",NERROR);
  printf("\nArchivo con sinapsis iniciales: %d.WTS",WTS); 
  printf("\nSemilla para la funcion rand(): %d",SEED); 
  if(CONTROL){
    printf("\nSINAPSIS:\n");
    for(j=1;j<=N2;j++) {
      for(i=0;i<=N1;i++) printf("%f\n",w1[j][i]);
    }
    for(j=1;j<=N3;j++) {
      for(i=0;i<=N2;i++) printf("%f\n",w2[j][i]);
    }   
  }

  return 0;
}
/* -------------------------------------------------------------------------------------- */
/*read_data: lee los datos de los archivos de entrenamiento(.data) y test(.test)
  filename es el nombre de los archivos (sin extension)
  La cantidad de datos y la estructura de los archivos fue leida en la funcion arquitec()
  Los registros en el archivo pueden estar separados por blancos ( o tab ) o por comas    */
/* -------------------------------------------------------------------------------------- */
int read_data(char *filename){

  FILE *fpat;
  float valor;
  int i,k,separador;

  sprintf(filepat,"%s.data",filename);
  fpat=fopen(filepat,"r");
  error=(fpat==NULL);
  if(error){
    printf("Error al abrir el archivo de datos\n");
    return 1;
  }

  if(CONTROL>1) printf("\n\nDatos de entrenamiento:");

  for(k=0;k<PTOT;k++){
	 if(CONTROL>1) printf("\nP%d:\t",k);
	 data[k][0]=-1.0;
 	 for(i=1;i<=N1+N3;i++){
	   fscanf(fpat,"%f",&valor);
	   data[k][i]=valor;
	   if(CONTROL>1) printf("%f\t",data[k][i]);
	   separador=getc(fpat);
	   if(separador!=',') ungetc(separador,fpat);
  	 }
  }
  fclose(fpat);

  if(!PTEST) return 0;

  sprintf(filepat,"%s.test",filename);
  fpat=fopen(filepat,"r");
  error=(fpat==NULL);
  if(error){
    printf("Error al abrir el archivo de test\n");
    return 1;
  }

  if(CONTROL>1) printf("\n\nDatos de test:");




  for(k=0;k<PTEST;k++){
	 if(CONTROL>1) printf("\nP%d:\t",k);
	 test[k][0]=-1.0;
         for(i=1;i<=N1+N3;i++){
	   fscanf(fpat,"%f",&valor);
	   test[k][i]=valor;
	   if(CONTROL>1) printf("%f\t",test[k][i]);
	   separador=getc(fpat);
	   if(separador!=',') ungetc(separador,fpat);
         }
  }


  fclose(fpat);


  fpat=fopen("phonems_stand.csv","r");
  error=(fpat==NULL);
  if(error){
    printf("Error al abrir articulatory features\n");
    return 1;
  }

  if(CONTROL>1) printf("\n\nArticulatory Features:");

  for(k=0;k<PHONEMS;k++){
 	 for(i=0;i<ARTICULATORY_FEATURES;i++){
	   fscanf(fpat,"%f",&valor);
	   afs[k][i]=valor;
	   if(1) printf("%f ",afs[k][i]);
	   separador=getc(fpat);
	   if(separador!=',') ungetc(separador,fpat);
  	 }
  	 if(1) printf("\n");

  }

  fclose(fpat);

  return 0;

}
/* ------------------------------------------------------------ */
/* shuffle: mezcla el vector seq al azar.
   El vector seq es un indice para acceder a los patrones.
   Los patrones mezclados van desde seq[0] hasta seq[hasta-1]
   Esto permite separar la parte de validacion de la de train   */
/* ------------------------------------------------------------ */
void shuffle(int hasta){
   float x;
   int tmp;
   int top,select;

   top=hasta-1;
   while (top > 0) {
	x = (float)rand();
	x /= (RAND_MAX+1.0);
	x *= (top+1);
	select = (int)x;
	tmp = seq[top];
	seq[top] = seq[select];
	seq[select] = tmp;
	top --;
   }
  if(CONTROL>3) {printf("End shuffle\n");fflush(NULL);}
}
/* ------------------------------------------------------------------- */
/*forward: propaga el vector de valores de entrada X1[] en la red
  En los vectores x2[] y x3[] quedan las activaciones correspondientes */
/* ------------------------------------------------------------------- */
void forward(){
  int i,j,k;
  float h;
  
  /*calcular los X2*/
  x2[0]=-1.0;
  for( j = 1 ;j<= N2;j++){
    h = 0.0;
    for( k = 0 ;k<= N1;k++) h += w1[j][k] * x1[k];
    x2[j] = sigmoid(h);
  }
  
  /*calcular los x3*/
  for( i = 1 ;i<= N3;i++){
    h = 0.0;
    for( j = 0 ;j<= N2;j++) h += w2[i][j] * x2[j];
    x3[i] = h ;  /* activacion lineal*/
  }
  if(CONTROL>3) {printf("End forward\n");fflush(NULL);}
}


float flabs(float a){
	return (a>0?a:-a);
}

float discrete_distance(float *float_prediction, float *v2){   //OK
	int dist=0;
	int i;
	for (i=0 ; i< ARTICULATORY_FEATURES ; i++){
		float tmp = float_prediction[i]>0.5f ? 1.0f : 0.0f;
		dist += flabs(tmp-v2[i]);
	}
	return dist;
}

int closestPhonem(float *m, int verb){  //OK
	// We assume that m is a prediction vector
	float minDis = 10000.0f;
	int minIx=-1;
	float temp;
	int i;

	if(verb){
		printf("Predicted phonem:");
		for (i=0 ; i< ARTICULATORY_FEATURES ; i++){
			printf("%f ",m[i]);
		}
		printf("\n");
	
		printf("Discretized predicted phonem:");
		for (i=0 ; i< ARTICULATORY_FEATURES ; i++){
			printf("%f ",m[i]>0.5f?1.0f:0.0f);
		}
		printf("\n");
	}
	for (i=0 ; i< PHONEMS ; i++){
		temp = discrete_distance(m,afs[i]);
		if (temp<minDis) { minDis = temp, minIx=i; }
		if(verb) printf("cP: Considering %d, dis %f \n",i,temp);
	}
	if(verb) printf("cP: The closest phonem is %d \n",minIx);
	return minIx;
}

int getRealPhonem(float *m){
	float target[N3];
	int i;
	printf("rP: The real phonem is: ");
	for (i=1 ; i<=N3 ; i++){
		target[i-1]=m[i+N1];
		printf("%f ",target[i-1]);
	}



	printf("\n");
	int ans = closestPhonem(target,0);
	printf("rP: The real phonem is %d \n",ans);

	return(ans);
}

/* ------------------------------------------------------------------------------ */
/*propagar: calcula los valores de salida de la red para un conjunto de datos
  la matriz S tiene que tener el formato adecuado ( definido en arquitec() )
  pat_ini y pat_fin son los extremos a tomar en la matriz
  usar_seq define si se accede a los datos directamente o a travez del indice seq
  los resultados (las propagaciones) se guardan en la matriz seq                  */
/* ------------------------------------------------------------------------------ */
float propagar(float **S,int pat_ini,int pat_fin,int usar_seq){

  float mse=0.0;
  int i,patron,nu;
  discrete_error=0.;
  
  for (patron=pat_ini; patron < pat_fin; patron ++) {

   /*nu tiene el numero del patron que se va a presentar*/
    if(usar_seq) nu = seq[patron];
    else         nu = patron;
        
    printf("Considering pattern %d of training \n",nu);
    /*cargar el patron en X1*/
    for (i = 0; i <= N1; i ++) x1[i] = S[nu][i];
    
    /*propagar la red*/
    forward();
    
    for(i = 1; i <= N3; i++) {
      /*generar matriz de predicciones*/
      pred[nu][i]=x3[i];
      /*actualizar error estimado*/
      mse += (x3[i]-S[nu][N1+i])*(x3[i]-S[nu][N1+i]);
    }   

	// CALCULATE DISCRETE ERROR
	int closest = closestPhonem(pred[nu]+1,1);
	int real    = getRealPhonem(S[nu]);
	printf("Real %d closest %d \n",real,closest);
	if (real!=closest) discrete_error+=1.0f;
 
  }

  mse /= ( (float)(pat_fin-pat_ini));
  discrete_error /= (float) (pat_fin-pat_ini);
  if(CONTROL>3) {printf("End prop\n");fflush(NULL);}
  return mse;
}
/* --------------------------------------------------------------------------------------- */
/*train: entrena la red
  Algoritmo: Stochastic Back propagation
  Las salidas son las curvas de entrenamiento (mse estocastico,mse al final de la epoca,
  mse validacion, mse test) en el archivo .mse y la matriz final de sinapsis en el archivo
  (WTS+1).wts
  Los resultados finales y la matriz resultante corresponden al minimo de validacion
  si no hay validacion, corresponde al minimo de mse al final de la epoca                   */
/* ---------------------------------------------------------------------------------------- */
int train(char *filename){

  FILE *ferror,*fpredic;

  int nu, nu1;
  int iter,epocas_del_minimo;
  int i,j,k;
  float h;
  float eta,suma;
  float mse,mse_train,mse_valid,mse_test,minimo_valid;
  float disc_train,disc_valid,disc_test;

  /* Inicializar archivos de control */
  sprintf(filepat,"%s.predic",filename);
  fpredic=fopen(filepat,"w");
  error=(fpredic==NULL);
  if(error){
    printf("Error al abrir archivo para guardar predicciones\n");
    return 1;
  }
  sprintf(filepat,"%s.mse",filename);
  ferror=fopen(filepat,"a");
  error=(ferror==NULL);
  if(error){
    printf("Error al abrir archivo para guardar curvas\n");
    return 1;
  }
  
  /* Inicializacion de todas las matrices */
  for (j = 1; j <= N2; j++) {
    grad2[j] = 0.0;
    for (k = 0; k <= N1; k++) dw1[j][k]=0.0;
  }
  for (i = 1; i <= N3; i ++) {
    grad3[i] =0.0;
    for (j = 0; j <= N2; j++) dw2[i][j]=0.0;
  }
  for(k=0;k<PTOT;k++) seq[k]=k;  /* inicializacion del indice de acceso a los datos */
  x1[0]=-1.0;                    /* bias de las unidades de cada hilera             */
  x2[0]=-1.0;
  minimo_valid=1000000.0;

  /* Fijar parametros: la nueva variable es para poder variar el learning rate durante el entrenamiento si se desea */
  eta=ETA;

  /*efectuar shuffle inicial de los datos de entrenamiento si SEED != -1*/
  if(SEED>-1){
    srand((unsigned)SEED);    
    shuffle(PTOT);
  }

  /* for principal: ITER iteraciones */
  for (iter=1; iter<=ITER; iter++) {
	printf("Iter %d \n",iter);
    mse = 0.0;

    shuffle(PR);

    /*barrido sobre los patrones de entrenamiento*/
    for(nu1=0;nu1<PR;nu1++) {

          nu = seq[nu1];  /*nu tiene el numero del patron que se va a presentar*/

	  /*cargar el patron en X1*/
	  for( k=1 ;k<=N1 ;k++) x1[k] = data[nu][k];
	  
	  /*propagar*/
	  forward();
	  
	  /*cargar los targets*/
	  for( k=1 ;k<=N3 ;k++) target[k] = data[nu][k+N1];

	  /*calcular gradiente en 3 hilera*/
	  for( i = 1 ;i<= N3;i++){
		 grad3[i] = (target[i] - x3[i]);  /*corresponde a lineal*/
	  }

	  /*calcular gradiente en 2 hilera*/
	  for( j = 1 ;j<= N2;j++){
		 suma = 0.0;
		 for( i = 1 ;i<= N3;i++)
			suma += grad3[i] * w2[i][j];
		 grad2[j] = suma * (1.0- x2[j]) * x2[j];
	  }

	  /*calcular dw2 y corregir w2*/
	  for( i = 1 ;i<= N3;i++) for( j = 0 ;j<= N2;j++){
		dw2[i][j]= u * dw2[i][j] + eta * grad3[i] * x2[j];
		w2[i][j] += dw2[i][j];
	  }

	  /*calcular dw1 y corregir w1*/
	  for( j = 1 ;j<= N2;j++) for( k = 0 ;k<= N1;k++){
		dw1[j][k]= u * dw1[j][k] + eta * grad2[j] * x1[k];
		w1[j][k] += dw1[j][k];
	  }

	  /*actualizar el mse*/
	  for( i = 1 ;i<= N3;i++){
		 mse += (x3[i] - target[i]) * (x3[i] - target[i]);
	  }

    }/* next nu1 - barrido sobre patrones*/


    /* controles: grabar error cada NERROR iteraciones*/
    if ((iter/NERROR)*NERROR == iter) {

      printf("Checkpoint! Iter %d \n",iter);
      	
      mse /= ((float)PR);               // BIEN  -> ERROR EN TRAIN DE LA ULTIMA ITERACION

      printf("Propagando en Training \n");

      //mse_train=propagar(data,0,PR,1);  // BIEN, PERO PARA MI DEBERIA DER LO MSIMO!
      //disc_train=discrete_error;        // MAL CALCULADO, ACA DEBERIA METER DEDO Y HACER QUE SE CALCULE BIEN 

      /*calcular mse de validacion; si no hay, usar mse_train*/
      
     /*
      if(PR==PTOT){
          mse_valid=mse_train;
          disc_valid=disc_train;
      }else{
          mse_valid=propagar(data,PR,PTOT,1);   // DEBERIA ESTAR BIEN CALCULADO...
          disc_valid=discrete_error;            // MAL CALCULADO!
      }
      //calcular mse de test (si hay)
      if (PTEST>0){
             mse_test =propagar(test,0,PTEST,0);
             disc_test=discrete_error;
      }else  mse_test = disc_test = 0.;
      fprintf(ferror,"%f\t%f\t%f\t%f\t",mse,mse_train,mse_valid,mse_test);
      fprintf(ferror,"%f\t%f\t%f\n",disc_train,disc_valid,disc_test);
      if(CONTROL) fflush(NULL);
      if(mse_valid<minimo_valid){
		sinapsis_save(WTS+1);
		minimo_valid=mse_valid;
		epocas_del_minimo=iter;
      }
      */
    }
    if(CONTROL>2) {printf("Iteracion %d\n",iter);fflush(NULL);}
    
  }/*next iter - lazo de iteraciones*/

  /*mostrar resumen del entrenamiento*/
  printf("\nFin del entrenamiento.\n\n");
  printf("Error final:\nEntrenamiento(est):%f\nEntrenamiento(med):%f\n",mse,mse_train);
  printf("Validacion:%f\nTest:%f\n",mse_valid,mse_test);
  
  /* Calcular y guardar predicciones sobre el archivo de test */
  /*leer pesos del minimo de validacion desde archivo*/
  sinapsis_read(WTS+1);
  mse_test=propagar(test,0,PTEST,0);
  disc_test=discrete_error;
  for(k=0; k < PTEST ; k++){
    for( i = 1 ;i<= N1;i++) fprintf(fpredic,"%d\t",(int)test[k][i]);
    for(i=1; i<=N3; i++)	fprintf(fpredic,"%d\t",pred[k][i]>0.5f?1:0);
    fprintf(fpredic,"\n");
  }
  printf("\nError minimo en validacion:\nEpoca:%d\nValidacion:%f\nTest:%f\nTest discreto:%f%%\n\n",epocas_del_minimo,minimo_valid,mse_test,100.0*disc_test);
  

  fclose(fpredic);
  fclose(ferror);

  return 0;

}


/* ----------------------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
int main(int argc, char **argv){

  if(argc!=2){
    printf("Modo de uso: bp <filename>\ndonde filename es el nombre del archivo (sin extension)\n");
    return 0;
  }

  /* defino la red e inicializo los pesos */
  error=arquitec(argv[1]);
  if(error){
    printf("Error en la definicion de la red\n");
    return 1;
  }

  /* leo los datos */
  error=read_data(argv[1]);                 
  if(error){
    printf("Error en la lectura de datos\n");
    return 1;
  }


  /* entreno la red */
  error=train(argv[1]);                     
  if(error){
    printf("Error en el entrenamiento\n");
    return 1;
  } 

  return 0;

}
/* ----------------------------------------------------------------------------------------------------- */












