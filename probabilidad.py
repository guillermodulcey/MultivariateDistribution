import math
import numpy
import random

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

class Probabilidad():
    def __init__(self):
        self.matrizCov = []
        pass

    def generarDataSet(self, mu:list, matrizCov:list, k, semilla):
        random.seed(semilla)
        datos = []
        datosR = []
        numVariables = len(mu)
        self.matrizCov = matrizCov
        for i in range(0,k):
            while True:
                z, vector = self.generarVector(numVariables)
                evaluacion = self.pdf(vector,mu,self.matrizCov)
                if z <= evaluacion:
                    datos.append(vector)
                    break
                else:
                    datosR.append(vector)

        return datos, datosR

    def generarVector(self,longitud):
        vector = []
        for i in range(0,longitud):
            vector.append(random.uniform(0,1))
        z = random.uniform(0,1)
        return z, vector
            
    def pdf(self, x:list, mu:list, matrixSigma):
        k = len(x)

        xmu = numpy.array(x) - numpy.array(mu)
        trans_xmu = xmu.transpose()

        nmatriz = numpy.array(matrixSigma)

        pi = math.pow(2*math.pi,(-k/2))

        determinante = numpy.linalg.det(nmatriz)
        det = math.pow(determinante,(-1/2))

        inv = numpy.linalg.inv(nmatriz)

        prodXmuInv = numpy.dot(trans_xmu,inv)
        prodFinal = numpy.dot(prodXmuInv,xmu)
        potencia = (-1/2)*prodFinal
        
        e = math.exp(potencia)
        return pi*det*e

    def generarMatriz(self, sigma:list, ro:list):
        matriz = []
        lFilas = len(sigma)
        contador = 0
        dr = list(map(self.decodificar, ro))
        for i in range(0,lFilas):
            fila = []
            for j in range(0,lFilas):
                if j>=i:
                    if i==j:
                        fila.append(sigma[i]*sigma[j])
                    else:
                        fila.append(dr[contador]*sigma[i]*sigma[j])
                        contador+=1
                else:
                    fila.append(0)
            matriz.append(fila)

        for i in range(0,lFilas-1):
            for j in range(i+1,lFilas):
                matriz[j][i] = matriz[i][j]

        return matriz

    #########Bivariate
    def pdfBivariate(self, x:list, mu:list, sigma:list, ro):
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
                    ro = random.uniform(0.8,1)
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

    def graficar(self,DataSet, DataSetR, mu):
        from matplotlib.ticker import LinearLocator
        xs = self.obtenerColumna(DataSet,0)
        ys = self.obtenerColumna(DataSet,1)
        zs = list(map(lambda a: 0, xs))

        xr = self.obtenerColumna(DataSetR, 0)
        yr = self.obtenerColumna(DataSetR, 1)
        zr = list(map(lambda a: 0, xr))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, color='green')

        ax.scatter(xr, yr, zr, color='red')
        #ax.set_zlim([0,1.1])

        z = []

        x = numpy.arange(0, 1, 0.01)
        y = numpy.arange(0, 1, 0.01)

        puntosFuncion = self.__generarMatrizFuncion()

        x,y = numpy.meshgrid(x,y)

        for i in range(0,len(puntosFuncion)):
            res = self.pdf(puntosFuncion[i],mu,self.matrizCov)
            z.append(res)

        z = numpy.array(z)
        z = z.reshape((len(x), len(y)))

        ax.plot_surface(y, x, z, rstride=8, cstride=8, alpha=0.8)

        plt.show()

    def __generarMatrizFuncion(self):
        datos = []
        x = numpy.arange(0, 1, 0.01).tolist()
        y = numpy.arange(0, 1, 0.01).tolist()
        for i in x:
            for j in y:
                datos.append([i,j])
        return datos

    def decodificar(self, x):
        if x == 'A':
            return random.uniform(0.8, 1)
        elif x == 'B':
            return random.uniform(-0.1, 0.1)
        elif x == 'I':
            return -random.uniform(0.8, 1)

    def archivo(self, DataSet:list):
        import csv

        with open('tabla.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(DataSet)


p = Probabilidad()

mu = [0,0]
sigma = [0.1,0.1]
#combinatoria(n,2) = longitud
ro = ["B","B","B"]
##############################
matrizCov = p.generarMatriz(sigma,ro)

# Semilla = 2. Funciona

DataSet, DataSetR = p.generarDataSet(mu,matrizCov,1000,2)

p.archivo(DataSet)

print(p.calcularPromedio(DataSet))
if len(mu)==2:
    p.graficar(DataSet, DataSetR,mu)
