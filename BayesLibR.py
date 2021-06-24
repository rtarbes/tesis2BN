# Configuración del ámbito de R sobre Python
import os
#os.environ['R_HOME'] = r'C:\Users\jtarb\anaconda3\envs\rBNLEARN\Lib\R'
#os.environ['R_PATH'] = r'C:\Users\jtarb\anaconda3\envs\rBNLEARN\Lib\R\bin\x64'

import BayesLibUtils as bnU
import bnlearn as bn
#import json as js
import sys
#import numpy as np

#from astropy.stats import *
#from scipy import stats
#from sklearn.metrics import *

#Librerias para el manejo de lenguaje R sobre Python
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.robjects import globalenv

bn1 = importr('bnlearn')
ut  = importr('utils')

#Librería para el graficado de los grafos DAG
from graphviz import Digraph

def AprendizajeR(model, fold, tipo, discreta=True, score="aic"):
    #Convirtiendo el DataFrame Pandas a un DataFrame en R
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_r = ro.conversion.py2rpy(model)
    
    #pasando el dataframe al ámbito R
    globalenv['df_r'] = df_r
    globalenv['puntuacion'] = score
    
    #transformando el dataframe a factores
    if discreta == True:
        r('df_r[] <- lapply(df_r, factor)')
    else:
        toFactor = r('c(1,2,3,4,23,24)')
        globalenv['toFactor'] = toFactor
        r('df_r[toFactor] <- lapply(df_r[toFactor], as.factor)')

        #cambiando algunas variables de int a num
        r('vector <- lapply(c(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22), function(i) as.numeric(df_r[,i]))')

        #reasignando los cambios de tipos
        y = 4
        for x in range(18):
            if x==18:
                y = 24
                globalenv['y'] = y
            else:
                y = y + 1
                globalenv['y'] = y
 
            globalenv['x'] = x+1
            r('df_r[, y]<-vector[x]')        
    
        #r('str(df_r)')
    
    #***********************************************
    #    Aprendiendo la estructura de los datos    *
    #***********************************************
    wl = r('matrix(c("tt", "game_score", "lt", "game_score", "pt", "game_score"), ncol = 2, byrow = TRUE)')
    bl = r('matrix(c("estado", "programa"), ncol = 2, byrow = TRUE)')
    globalenv['wl'] = wl
    globalenv['bl'] = bl
    dag = r('hc(na.omit(df_r), score=puntuacion)')
    #dag = r('hc(na.omit(df_r), whitelist=wl, blacklist = bl, score=puntuacion)')
    #pasando la estructura aprendida DAG al ambito R
    globalenv['dag'] = dag  
    
    #guardando la estructura estructura aprendida
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
    
    #**************************************************
    #aprendiendo los parametros a priori del modelo.
    #**************************************************
    pdag = r('bn.fit(dag, data = df_r, method = "mle")')
    #pasando los parametros aprendidos al ámbito R
    globalenv['pdag'] = pdag

    #guardando la estructura de parametros aprendida
    nombreArchivo = 'ParametrosCPD_'+tipo+'_'+str(fold)
    nombreArchivoExt = 'ExperimentosR\\'+nombreArchivo+'.gv'
    f = Digraph(name=nombreArchivo, filename=nombreArchivoExt, format='pdf', engine='dot', encoding="utf-8")

    f.attr('node', shape='circle')
    for x in range(len(pdag)):
        f.node(str(pdag[x][0][0]))
    
    f.attr('node', shape='circle')
    for x in range(len(pdag)): #recorriendo los nodos 
        l = pdag[x][2]

        for y in range(len(l)): #recorriendo los hijos
            xfrom = pdag[x][0] 
            xto   = pdag[x][2][y]
        
            f.edge(str(xfrom[0]), str(xto))    
    
    f.save()
    f.render(filename=nombreArchivoExt, view=False, cleanup=1)   
    
    return pdag

def probabilidadConjuntaR(model, test, fold, tipo, discreta):
    
    #Convirtiendo el DataFrame Pandas a un DataFrame en R
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_test = ro.conversion.py2rpy(test)
    
    #pasando el dataframe al ámbito R
    globalenv['df_test'] = df_test
    globalenv['pdag'] = model
    
    #transformando el dataframe a factores
    if discreta == True:
        r('df_test[] <- lapply(df_test, factor)')
    else:
        toFactor = r('c(1,2,3,4,23,24)')
        globalenv['toFactor'] = toFactor
        r('df_test[toFactor] <- lapply(df_test[toFactor], as.factor)')

        #cambiando algunas variables de int a num
        r('vector <- lapply(c(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22), function(i) as.numeric(df_test[,i]))')

        #reasignando los cambios de tipos
        y = 4
        for x in range(18):
            if x==18:
                y = 24
                globalenv['y'] = y
            else:
                y = y + 1
                globalenv['y'] = y
 
            globalenv['x'] = x+1
            r('df_test[, y]<-vector[x]')        
    
        #r('str(df_r)')
        
    #**************************************************
    # Calculando la probabilidad Conjunta (Inferencia)
    #**************************************************
    #obteniendo el numero de registros del dataframe
    x = r('nrow(df_test)')
    y = r('ncol(df_test)')
    
    #se define un arreglo unidimensional para registrar el resultado de la inferencia
    arreglo = []
    indice  = 0

    #recorriendo el data frame para calcular la probabilidad conjunta
    for i in range(x[0]):
        globalenv['i'] = i+1

        #construyendo la linea de evidencia a calcular
        if discreta == True:
            str1 = r('paste(colnames(df_test[-24]), "=", shQuote(sapply(df_test[i,-24], as.character), type="cmd"), collapse=",")')
        else:
            r('sPbc <- NULL')
            for j in range(y[0]-2,-1,-1):
                globalenv['j'] = j+1
                
                r('colName <- colnames(df_test[j])')
                tipoDato = r('sapply(colName, function(x) class(df_test[[x]]))')
                #print(tipo[0])
                if tipoDato[0] == 'factor':
                    r('colValor <- shQuote(df_test[i, j], type="cmd")')
                else:
                    r('colValor <- df_test[i, j]')
                
                r('sPbc <- paste(colName, "=", colValor, "," ,sPbc)')
            
            str1 = r('gsub(".{2}$", "", sPbc)')
            

        #obteniendo la evidencia de la variable "estado"
        ev = r('as.character(df_test[i, 24])')
        ev = '"' + ev[0] + '"'
        globalenv['ev'] = ev
        str2 = r('paste("(", colnames(df_test[24]), " == ", ev, ")", sep = "") ') 

        regSalida = "FILA N°: " + str(i+1) + " -> "+"P" + str2[0] + " | (" + str1[0] +")\n"
        print(regSalida)
    
        #construyendo la condición de evidencia con la APROBACION
        evA = '"' + '0' + '"' #Aprobado == 0"
        globalenv['evA'] = evA
        strA = r('paste("(", colnames(df_test[24]), " == ", evA, ")", sep = "") ') 
        #construyendo la condición de evidencia con la REPROBACION
        evR = '"' + '1' + '"' #Reprobado == 1
        globalenv['evR'] = evR
        strR = r('paste("(", colnames(df_test[24]), " == ", evR, ")", sep = "") ') 

        #pasando las variables al ámbito R
        globalenv['str1'] = "("+str1[0]+")"
        globalenv['strA'] = strA[0]
        globalenv['strR'] = strR[0]

        #Indicando el metodo a usar: Likelihood Weighting (Ponderación de probabilidad)
        me = '"lw"'
        globalenv['mt'] = me

        arreglo.append([])

        #calculando la probabilidad conjunta de la aprobacion y reprobación.
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

        #Extrayendo la clase con probabilidad mas alta
        if (pbcA[0] > pbcR[0]):
            arreglo[indice].append(0)
            arreglo[indice].append(pbcA[0])
        else:
            arreglo[indice].append(1)
            arreglo[indice].append(pbcR[0])

        #Guardando las metricas en un archivo
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
