import math
import numpy
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Probabilidad():
    def __init__(self):
        pass

    def generarDataSet(self, mu:list, sigma:list, k, semilla):
        random.seed(semilla)
        datos = []
        numVariables = len(mu)
        ro = self.__generarRo(numVariables)
        matrizCov = self.generarMatriz(sigma,ro)
        print(matrizCov) 
        for i in range(0,k):
            while True:
                z, vector = self.generarVector(numVariables)
                evaluacion = self.pmf(vector,mu,matrizCov)
                if z <= evaluacion:
                    datos.append(vector)
                    break
        return datos

    def generarVector(self,longitud):
        vector = []
        for i in range(0,longitud):
            vector.append(random.uniform(0,1))
        z = random.uniform(0,1)
        return z, vector
            
    def pmf(self, x:list, mu:list, matrixSigma):
        k = len(x)

        xmu = numpy.array(x) - numpy.array(mu)
        trans_xmu = xmu.transpose()

        nmatriz = numpy.array(matrixSigma)

        pi = math.pow(2*math.pi,(k/2))

        ####Revisar####
        determinante = numpy.linalg.det(nmatriz)
        det = math.pow(determinante,(1/2))
        ###############

        inv = numpy.linalg.inv(nmatriz)

        prodXmuInv = numpy.dot(trans_xmu,inv)
        prodFinal = numpy.dot(prodXmuInv,xmu)
        potencia = (-1/2)*prodFinal
        
        e = math.exp(potencia)
        return (1/(pi*det))*e

    def generarMatriz(self, sigma:list, ro):
        matriz = []
        lFilas = len(sigma)
        for i in range(0,lFilas):
            fila = []
            for j in range(0,lFilas):
                if i==j:
                    fila.append(sigma[i]*sigma[j])
                else:
                    fila.append(ro[i][j]*sigma[i]*sigma[j])
            matriz.append(fila)

        for i in range(0,lFilas):
            for j in range(0,lFilas):
                matriz[i][j] = matriz[j][i]

        return matriz

    #########Bivariate
    def pmfBivariate(self, x:list, mu:list, sigma:list, ro):
        resta = 1-math.pow(ro,2)
        raiz = math.pow(resta,(1/2))
        coeficiente = 2*math.pi*sigma[0]*sigma[1]

        parte1 = 1/coeficiente
        potencia = -(1/(2*resta))*(self.__cuadradosP(x[0],mu[0],sigma[0])-(2*ro*self.__productoP(x,mu,sigma))+self.__cuadradosP(x[1],mu[1],sigma[1]))
        parte2 = math.exp(potencia)
        return parte1*parte2
    
    def __cuadradosP(self,x,mu,sigma):
        expresion = (x-mu)/sigma
        return math.pow(expresion,2)

    def __productoP(self, x:list, mu:list, sigma:list):
        return (x[0]-mu[0])*(x[1]-mu[1])/(sigma[0]*sigma[1])

    def __generarRo(self, size):
        roMatriz = []
        for i in range(0,size):
            roFila = []
            for j in range(0,size):
                if i==j:
                    ro = 1
                else:
                    ro = random.uniform(0,1)
                roFila.append(ro)
            roMatriz.append(roFila)

        return roMatriz
    
    def calcularPromedio(self, matriz):
        variables = len(matriz[0])
        acumulador = []
        for i in range(0,variables):
            acumulador.append(0)

        for i in range(0,len(matriz)):
            for j in range(0,variables):
                acumulador[j] += matriz[i][j]

        return list(map(lambda a: a/len(matriz),acumulador))

    def obtenerColumna(self, matriz, column):
        return numpy.array(matriz)[:,column].tolist()


p = Probabilidad()
DataSet = p.generarDataSet([0.5,0.8],[0.1,0.1],1000,2)
print(DataSet)
print(p.calcularPromedio(DataSet))
xs = p.obtenerColumna(DataSet,0)
ys = p.obtenerColumna(DataSet,1)

plt.plot(xs, ys,'ro')
plt.show()
