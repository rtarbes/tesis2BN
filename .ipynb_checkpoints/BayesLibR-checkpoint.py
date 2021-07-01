"""BayesLibR. Proporciona las herramientas necesarias para la calcular los modelos Bayesianos en R."""
# ------------------------------------------------
# Name        : BayesLibR.py
# Authors     : Juan Tarbes Vergara
#               Pamela Morales Vergara
# Contact     : j.tarbesvergara@uandresbello.edu
#               p.moralesvergara2@uandresbello.edu
# Licence     : Universidad Andrés Bello (Chile)
# ------------------------------------------------

# %% Librerias
# Libreria para el manejo de archivos
import os

# En caso de que el ambiente R no tenga las variables de sistema bien configuradas
# ubicar el directorio de instalación de R en su computadora y asignarlo a las 
# variables R_HOME y R_PATH como se describe en las siguientes dos lineas.
# Ejemplo:
#    os.environ['R_HOME'] = r'C:\Users\<usr>\anaconda3\envs\rBNLEARN\Lib\R'
#    os.environ['R_PATH'] = r'C:\Users\<usr>\anaconda3\envs\rBNLEARN\Lib\R\bin\x64'

# Librería propietaria utilitaria
import BayesLibUtils as bnU

# Libreria BNLEARN es un paquete de Python para aprender la estructura gráfica de redes bayesianas, 
# aprendizaje de parámetros, métodos de inferencia y muestreo. 
import bnlearn as bn

# Libreria para el manejo de excepciones en Python
import sys

#Librerias para el manejo de lenguaje R sobre Python
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.robjects import globalenv

# libreria BNLEARN que es un paquete de R para aprender la estructura gráfica de las redes bayesianas, estimar sus parámetros 
# y realizar algunas inferencias útiles
bn1 = importr('bnlearn')

#ut  = importr('utils')

# Librería para graficar los grafos (DAG)
from graphviz import Digraph


def AprendizajeR(model, fold, tipo, discreta, score, clase):
    """ Aprendizaje de estructura y parámetros de BNLEARN en R
    
    Descripción
    -----------
    Encapsula el aprendizaje de estructura y el aprendizaje de parámetros (probabilidad anterior).
    El aprendizaje de la estructura es guardado en un archivo PDF en la carpeta \Experimentos con el nombre 
    "EstructuraCPD_'tipo'_fold.gv.pdf"
        
    Parámetros
    ----------
    model: recibe un dataframe con los la información a aprender 
    fold : Valor númerico que indica el cross-validation del cual se está solicitando el aprendizaje
    
    tipo :  Indicador del tipo de datos del que se desa aprender.  Valores posibles:
        a. "TRAIN"
        b. "TEST"
    
    discreta: {True, False}
        a. True  -> Le indica que todas las variables serán factores (discretas)
        b. False -> Le indica que las variables serán mixtas (discretas y continuas)

    score: Tipo de score a utilizar para el aprendizaje de la estructura
        a. "aic"       -> Criterio de información de Akaike. Solo para variables discretas, es decir, cuando flag "discreta = True"
        b. "bic"       -> Criterio de información Bayesiano. Solo para variables discretas, es decir, cuando flag "discreta = True"
        c. "loglik"    -> Probabilidad logarítmica. Solo para variables discretas, es decir, cuando flag "discreta = True"
        d. "aic-cg"    -> Solo para variables mixtas, es decir, cuando flag "discreta = False"
        e. "bic-cg"    -> Solo para variables mixtas, es decir, cuando flag "discreta = False"
        f. "loglik-cg" -> Solo para variables mixtas, es decir, cuando flag "discreta = False"
         
    Retorna
    -------
    Retorna un modelo aprendido "pdag"
       
    """    

    # Convirtiendo el DataFrame Pandas a un DataFrame en R
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_r = ro.conversion.py2rpy(model)
    
    # Pasando el dataframe al ámbito R
    globalenv['df_r'] = df_r
    globalenv['puntuacion'] = score
    globalenv['clase'] = clase
    
    r('xclase = c("estado")')
   
    # Transformando el dataframe a factores o Factores y Numéricos
    if discreta == True:
        r('df_r[] <- lapply(df_r, factor)')
    else:
        for i in range(len(df_r)):
            globalenv['i'] = i+1
            xyz = r('is.integer(df_r[,i])')
            
            if xyz[0] == False:
                r('df_r[i] <- lapply(df_r[i], as.factor)')
            else:
                r('df_r[i] <- lapply(df_r[i], as.numeric)')
      
        r('df_r[names(df_r) %in% xclase] <- lapply(df_r[names(df_r) %in% xclase], as.factor)')

    # -------------------------------------------
    #    Aprendiendo la estructura de los datos -
    # -------------------------------------------
    #wl = r('matrix(c("tt", "game_score", "lt", "game_score", "pt", "game_score"), ncol = 2, byrow = TRUE)')
    #bl = r('matrix(c("estado", "programa"), ncol = 2, byrow = TRUE)')
    #globalenv['wl'] = wl
    #globalenv['bl'] = bl
    #dag = r('hc(na.omit(df_r), whitelist=wl, blacklist = bl, score=puntuacion)')
    dag = r('hc(na.omit(df_r), score=puntuacion)')
    
    # Pasando la estructura aprendida DAG al ambito R
    globalenv['dag'] = dag  
    
    # Guardando la estructura estructura aprendida
    nombreArchivo = 'EstructuraCPD_'+tipo+'_'+str(fold)
    nombreArchivoExt = 'ExperimentosR\\'+nombreArchivo+'.gv'
    f = Digraph(name=nombreArchivo, filename=nombreArchivoExt, format='pdf', engine='dot', encoding="utf-8")

    f.attr('node', shape='circle')
    for x in range(len(dag[1].names)):
        f.node(dag[1].names[x])

    f.attr('node', shape='circle')
    for y in range(dag[2].nrow):
        xfrom = dag[2].rx(y+1,1).rx(1)[0]
        xto   = dag[2].rx(y+1,2).rx(1)[0]
        f.edge(xfrom, xto)

    #f.view()
    f.save()
    f.render(filename=nombreArchivoExt, view=False, cleanup=1)    
    
    # ------------------------------------------------
    # Aprendiendo los parametros a priori del modelo -
    # ------------------------------------------------
    pdag = r('bn.fit(dag, data = df_r, method = "mle")')
    
    # Pasando los parametros aprendidos al ámbito R
    globalenv['pdag'] = pdag
    
    return pdag

def probabilidadConjuntaR(model, test, fold, tipo, discreta, clase):
    """ Probabilidad conjunta (inferencia) de BNLEARN de R
    
    Descripción
    -----------
    Realiza el calculo de la probabilidad conjunta (inferencia)
        
    Parámetros
    ----------
    model: recibe un modelo DAG con los los parámetros aprendidos (probabilidad anterior)
    test : Data Frame con los datos con los cuales se construira la evidencia para la probabilidad conjunta.
    fold : Valor númerico que indica el cross-validation del cual se está solicitando el aprendizaje
    
    tipo :  Indicador del tipo de datos del que se desa aprender.  Valores posibles:
        a. "TRAIN"
        b. "TEST"
    
    discreta: {True, False}
        a. True  -> Le indica que todas las variables serán factores (discretas)
        b. False -> Le indica que las variables serán mixtas (discretas y continuas)

    clase: nombre de la variabla "clase" que se está calculando
         
    Retorna
    -------
    Retorna una matriz con los los valores de inferencia máximos de cada registro del dataframe "parámetro: test"
       
    """    
    
    # Convirtiendo el DataFrame Pandas a un DataFrame en R
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_test = ro.conversion.py2rpy(test)
    
    # Pasando el dataframe al ámbito R
    globalenv['df_test'] = df_test
    globalenv['pdag'] = model
    #globalenv['clase'] = "\'"+clase+"\'"

    r('xclase = c("estado")')
    print(r('xclase'))
    # Transformando el dataframe a factores
    if discreta == True:
        r('df_test[] <- lapply(df_test, factor)')
    else:
        for i in range(len(df_test)):
            globalenv['i'] = i+1
            
            xyz = r('is.integer(df_test[,i])')
            
            if xyz[0] == False:
                r('df_test[i] <- lapply(df_test[i], as.factor)')
            else:
                r('df_test[i] <- lapply(df_test[i], as.numeric)')
        
        r('df_test[names(df_test) %in% xclase] <- lapply(df_test[names(df_test) %in% xclase], as.factor)')
        
    # --------------------------------------------------
    # Calculando la probabilidad Conjunta (Inferencia) -
    # --------------------------------------------------
    
    # Obteniendo el numero de registros del dataframe
    x = r('nrow(df_test)')
    y = r('ncol(df_test)')
    
    # Se define un arreglo unidimensional para registrar el resultado de la inferencia
    arreglo = []
    indice  = 0

    # Recorriendo el data frame para calcular la probabilidad conjunta
    for i in range(x[0]):
        globalenv['i'] = i+1
       
        # Construyendo la linea de evidencia a calcular
        if discreta == True:
            str1 = r('paste(colnames(df_test[!(names(df_test) %in% xclase)]), "=", shQuote(sapply(df_test[i, !(names(df_test) %in% xclase)], as.character), type="cmd"), collapse=",")')
        else:
            r('sPbc <- NULL')
            for j in range(y[0]-2,-1,-1):
                globalenv['j'] = j+1
                
                r('colName <- colnames(df_test[j])')
                tipoDato = r('sapply(colName, function(x) class(df_test[[x]]))')
                
                if tipoDato[0] == 'factor':
                    r('colValor <- shQuote(df_test[i, j], type="cmd")')
                else:
                    r('colValor <- df_test[i, j]')
                
                r('sPbc <- paste(colName, "=", colValor, "," ,sPbc)')
            
            str1 = r('gsub(".{2}$", "", sPbc)')

        # Obteniendo la evidencia de la variable "xclase"
        ev = r('as.character(df_test[i, (names(df_test) %in% xclase)])')
        ev = '"' + ev[0] + '"'
        globalenv['ev'] = ev
        str2 = r('paste("(", colnames(df_test[names(df_test) %in% xclase]), " == ", ev, ")", sep = "") ') 

        regSalida = "FILA N°: " + str(i+1) + " -> "+"P" + str2[0] + " | (" + str1[0] +")\n"
        print(regSalida)
    
        # Construyendo la condición de evidencia con la APROBACION
        evA = '"' + '0' + '"' # Aprobado == 0"
        globalenv['evA'] = evA
        strA = r('paste("(", colnames(df_test[names(df_test) %in% clase]), " == ", evA, ")", sep = "") ') 
        
        # Construyendo la condición de evidencia con la REPROBACION
        evR = '"' + '1' + '"' # Reprobado == 1
        globalenv['evR'] = evR
        strR = r('paste("(", colnames(df_test[names(df_test) %in% clase]), " == ", evR, ")", sep = "") ') 

        # Pasando las variables al ámbito R
        globalenv['str1'] = "("+str1[0]+")"
        globalenv['strA'] = strA[0]
        globalenv['strR'] = strR[0]

        # Indicando el metodo a usar: Likelihood Weighting (Ponderación de probabilidad)
        me = '"lw"'
        globalenv['mt'] = me

        arreglo.append([])

        # Calculando la probabilidad conjunta de la aprobacion y reprobación.
        try:
            pbcA = r('eval(parse(text=paste("cpquery(pdag, event =",strA,", evidence = list",str1,",n=10^4, method=",mt,")")))')
        except:
            pbcA = [0]
            
        try:
             pbcR = r('eval(parse(text=paste("cpquery(pdag, event =",strR,", evidence = list",str1,",n=10^4, method=",mt,")")))')
        except:
            pbcR = [0]
            
        print("A: " + str(pbcA[0]))
        print("R: " + str(pbcR[0])+"\n")    

        # Extrayendo la clase con probabilidad mas alta
        if (pbcA[0] > pbcR[0]):
            arreglo[indice].append(0)
            arreglo[indice].append(pbcA[0])
        else:
            arreglo[indice].append(1)
            arreglo[indice].append(pbcR[0])

        # Guardando las metricas en un archivo
        filename =  'ExperimentosR\\'+'ProbConjunta_'+tipo+'_'+str(fold)+'.txt'
        if os.path.exists(filename): 
            file = open(filename, "a")
        else:
            file = open(filename, "w")

        file.write('('+tipo+') Fold ' + str(fold) + ': ' + regSalida)  
        file.write("A: " + str(pbcA[0])+"\n")
        file.write("R: " + str(pbcR[0])+"\n")    
        file.write("\n")
        file.close()

        indice += 1
            
    return arreglo   
