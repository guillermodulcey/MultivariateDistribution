import math
import numpy
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Probabilidad():
    def __init__(self):
        self.matrizCov = []
        pass

    def generarDataSet(self, mu:list, sigma:list, k, semilla):
        random.seed(semilla)
        datos = []
        numVariables = len(mu)
        ro = self.__generarRo(numVariables)
        self.matrizCov = self.generarMatriz(sigma,ro)
        print(self.matrizCov) 
        for i in range(0,k):
            while True:
                z, vector = self.generarVector(numVariables)
                evaluacion = self.pmf(vector,mu,self.matrizCov)
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

        pi = math.pow(2*math.pi,(-k/2))

        ####Revisar####
        determinante = numpy.linalg.det(nmatriz)
        det = math.pow(determinante,(-1/2))
        ###############

        inv = numpy.linalg.inv(nmatriz)

        prodXmuInv = numpy.dot(trans_xmu,inv)
        prodFinal = numpy.dot(prodXmuInv,xmu)
        potencia = (-1/2)*prodFinal
        
        e = math.exp(potencia)
        return pi*det*e

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
                    ro = random.uniform(0.1,0.3)
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

    def graficar(self, xs, ys, zs, mu):
        from matplotlib.ticker import LinearLocator
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs)
        #ax.set_zlim([0,1])

        puntosFuncion = self.__generarMatrizFuncion()

        z = []

        x = numpy.arange(0, 1, 0.01)
        y = numpy.arange(0, 1, 0.01)

        x,y = numpy.meshgrid(x,y)
        print(x)
        
        for i in range(0,len(puntosFuncion)):
            res = self.pmf(puntosFuncion[i],mu,self.matrizCov)
            z.append(min(1,res))

        for i in range(0,len(puntosFuncion)):
            if z[i]>1:
                print(x[i],y[i],z[i])

        z = numpy.array(z)
        z = z.reshape((len(x), len(y)))
        ax.contour3D(x, y, z, 50, cmap='binary')


        plt.show()
    
    def __generarMatrizFuncion(self):
        datos = []
        x = numpy.arange(0, 1, 0.01).tolist()
        y = numpy.arange(0, 1, 0.01).tolist()
        for i in x:
            for j in y:
                datos.append([i,j])
        return datos

    def graficarEnsayo(self):
        from matplotlib.ticker import LinearLocator
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = numpy.arange(-5, 5, 0.25)
        xlen = len(X)
        Y = numpy.arange(-5, 5, 0.25)
        ylen = len(Y)
        X, Y = numpy.meshgrid(X, Y)
        R = numpy.sqrt(X**2 + Y**2)
        Z = numpy.sin(R)

        # Create an empty array of strings with the same shape as the meshgrid, and
        # populate it with two colors in a checkerboard pattern.
        colortuple = ('y', 'b')
        colors = numpy.empty(X.shape, dtype=str)
        for y in range(ylen):
            for x in range(xlen):
                colors[x, y] = colortuple[(x + y) % len(colortuple)]

        # Plot the surface with face colors taken from the array we made.
        surf = ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0)

        # Customize the z axis.
        ax.set_zlim(-1, 1)
        ax.w_zaxis.set_major_locator(LinearLocator(6))

        plt.show()


p = Probabilidad()
mu = [0,0]
DataSet = p.generarDataSet(mu,[0.1,0.1],1000,2)
#print(DataSet)
#print(p.calcularPromedio(DataSet))

xs = p.obtenerColumna(DataSet,0)
ys = p.obtenerColumna(DataSet,1)
zs = list(map(lambda a: 0, xs))

p.graficar(xs,ys,zs,mu)
#p.graficarEnsayo()