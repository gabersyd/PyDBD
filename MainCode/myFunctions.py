import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.sparse.linalg as la
import scipy.sparse as sparse
import time as tm
import logging


####==============   POISSON MATRIX USED TO SOLVE THE POISSON'S EQUATION IMPLICTLY   ========================
#============================================================================================================
def SparseLaplacianOperator(nx,k1=-1,k2=0,k3=1):
	#this function simply creates a tridiagonal matrix with -2 on the main diagonal and
	#1 on the diagonal below and above the main diagonal. nx is the size of the matrix
	d1 = np.zeros((nx),float)
	d2 = np.ones((nx),float)
	d3 = np.zeros((nx),float)
	d1[:-2] = 1; d2[1:-1]=-2;d3[2:]=1
	return (sparse.dia_matrix(([d1,d2,d3],[k1,k2,k3]),shape=(nx,nx)).tocsc() ) 
#------------------------------------------------------------------------------------------------------------


def explicitdiffusionoperator(inputdataa,diffusiondata,dx,dt):
	kvalue = 0.5*(diffusiondata[:-1]+diffusiondata[1:])
	return inputdataa[1:-1]+(dt/dx**2)*(kvalue[1:]*(inputdataa[2:]-inputdataa[1:-1])-kvalue[:-1]*(inputdataa[1:-1]-inputdataa[:-2]))
	#return inputdataa[1:-1]+(dt/dx**2)*(diffusiondata[2:]*(inputdataa[2:]-inputdataa[1:-1])-diffusiondata[:-2]*(inputdataa[1:-1]-inputdataa[:-2]))
	#return inputdataa[1:-1]+(dt/dx**2)*(0*(diffusiondata[2:]-diffusiondata[:-2])*(inputdataa[2:]-inputdataa[:-2])+diffusiondata[1:-1]*(inputdataa[2:]-2*inputdataa[1:-1]+inputdataa[:-2]))


def diffusionfct(ns,ngrid0,inputdataa,diffusiondata,dx,dt):
	densityvalue = np.zeros((ns,ngrid0+2+4),float)
	diffusionvalue = np.zeros((ns,ngrid0+2+4),float)
	densityvalue[:,2:-2] = inputdataa
	diffusionvalue[:,2:-2] = diffusiondata
	densityvalue[:,-2] = densityvalue[:,-1]=densityvalue[:,-3]
	densityvalue[:,0] = densityvalue[:,1]=1*densityvalue[:,2]
	diffusionvalue[:,0] = diffusionvalue[:,1]=diffusionvalue[:,2]
	diffusionvalue[:,-1] = diffusionvalue[:,-2]=diffusionvalue[:,-3]
	flow = -dt*0.5*(diffusionvalue[:,:-1]+diffusionvalue[:,1:])*(densityvalue[:,1:]-densityvalue[:,:-1])/dx
	atd = densityvalue[:,1:-1]-(flow[:,1:]-flow[:,:-1])/dx
	averagepart = 0.5*(densityvalue[:,1:]+densityvalue[:,:-1])
	fvalueo = (averagepart[:,1:]-averagepart[:,:-1])*diffusionvalue[:,1:-1]/dx
	fhigh = -dt*((7/12)*(fvalueo[:,2:-1]+fvalueo[:,1:-2])-(1/12)*(fvalueo[:,3:]+fvalueo[:,:-3]))
	adif = fhigh[:]-flow[:,2:-2]
	signmatrix = (adif>=0)*2.-1
	AC = signmatrix*np.maximum(0,np.minimum(np.abs(adif),np.minimum(signmatrix*dx*(atd[:,3:]-atd[:,2:-1]),signmatrix*dx*(atd[:,1:-2]-atd[:,:-3]))))
	return atd[:,2:-2]-(AC[:,1:]-AC[:,:-1])/dx
	
	
def diffusionfctE(ngrid0,inputdataa,diffusiondata,dx,dt):
	densityvalueE = np.zeros((ngrid0+2+4),float)
	diffusionvalueE = np.zeros((ngrid0+2+4),float)
	densityvalueE[2:-2] = inputdataa
	diffusionvalueE[2:-2] = diffusiondata
	densityvalueE[-2:] = 1*densityvalueE[-3]
	densityvalueE[0:2] = 1*densityvalueE[2]
	diffusionvalueE[0:2] = 1*diffusionvalueE[2]
	diffusionvalueE[-2:] = 1*diffusionvalueE[-3]
	flow = -dt*0.5*(diffusionvalueE[:-1]+diffusionvalueE[1:])*(densityvalueE[1:]-densityvalueE[:-1])/dx
	atd = densityvalueE[1:-1]-(flow[1:]-flow[:-1])/dx
	averagepart = 0.5*(densityvalueE[1:]+densityvalueE[:-1])
	fvalueo = (averagepart[1:]-averagepart[:-1])*diffusionvalueE[1:-1]/dx
	fhigh = -dt*((7/12)*(fvalueo[2:-1]+fvalueo[1:-2])-(1/12)*(fvalueo[3:]+fvalueo[:-3]))
	adif = fhigh-flow[2:-2]
	signmatrix = (adif>=0)*2.-1
	AC = signmatrix*np.maximum(0,np.minimum(np.abs(adif),np.minimum(signmatrix*dx*(atd[3:]-atd[2:-1]),signmatrix*dx*(atd[1:-2]-atd[:-3]))))
	return (atd[2:-2]-(AC[1:]-AC[:-1])/dx)

	

#SPARSE TRIDIAGONAL MATRIX USED TO SOLVE THE DIFFUSION EQUATION IMPLICTLY====================================
#============================================================================================================
def SparseDiffusionOperator(numberdensity,dif,dx,dt,k1=-1,k2=0,k3=1):
	nx = dif.size
	d1 = np.zeros((nx),float)  #diagonal below main diagonal
	d2 = np.ones((nx),float)   #main diagonal
	d3 = np.zeros((nx),float)  #diagonal above main diagonal
	d1[:-2] = (dt/(4*dx*dx))*(dif[2:]-dif[:-2]-4*dif[1:-1])
	d2[1:-1] = (1+2*dt*dif[1:-1]/(dx**2))
	d3[2:] = (dt/(4*dx*dx))*(-dif[2:]+dif[:-2]-4*dif[1:-1])
	return (la.spsolve((sparse.dia_matrix(([d1,d2,d3],[k1,k2,k3]),shape=(nx,nx)).tocsc()),numberdensity))
#------------------------------------------------------------------------------------------------------------


#ADVECTION SOLVING MATRIX (1D FINITE VOLUME METHOD) ==========================================================
def AdvectionAlgorithm(dx,dt,velocity,density):
	#this is a simple yet very efficient code that solves the advection of the particles using FVM
	flux = (0.5*(velocity[:,1:]*density[:,1:]+velocity[:,:-1]*density[:,:-1])-
		  0.5*0.5*abs(velocity[:,1:]+velocity[:,:-1])*(density[:,1:]-density[:,:-1]))*dt
	density[:,1:-1] += -(flux[:,1:]-flux[:,:-1])/dx
	return density[:,1:-1]
#------------------------------------------------------------------------------------------------------------


#ADVECTION SOLVING MATRIX (1D FINITE VOLUME METHOD) ==========================================================
def AdvectionAlgorithmE(dx,dt,velocity,density):
	#this is a simple yet very efficient code that solves the advection of the particles using FVM
	flux = (0.5*(velocity[1:]*density[1:]+velocity[:-1]*density[:-1])-
		  0.5*0.5*abs(velocity[1:]+velocity[:-1])*(density[1:]-density[:-1]))*dt
	density[1:-1] += -(flux[1:]-flux[:-1])/dx
	return density[1:-1]


def importtransportdiffusion():
	#------ Importing mobility and diffusion as a function of efield---------------
	#==============================================================================
	#these lines of code are simply used to import the values of transport coefficient as a function of electric field	 from a text file tableEfield.txt
	parameterSize = 999
	importfile = np.loadtxt('table/tableEfield.txt',dtype=6*'float,'+'float',delimiter='\t',usecols=list(range(7)),skiprows=1,unpack=True)
	mobilityInput = np.zeros((4,parameterSize),float)
	diffusionInput = np.zeros((4,parameterSize),float)
	mobilityInput[0,:] = np.array(importfile[0]);mobilityInput[1,:]=np.array(importfile[1]);mobilityInput[2,:]=np.array(importfile[2])
	diffusionInput[0,:] = np.array(importfile[3]);diffusionInput[1,:]=np.array(importfile[4]);diffusionInput[2,:]=np.array(importfile[5]);diffusionInput[3,:]=np.array(importfile[6])
	#------------ Importing reaction rates as a function of efield-----------------
	#==============================================================================
	#these lines of code are simply used to import the values of transport coefficient as a function of electron energy from a text file tableEnergy.txt
	npoints = 190
	energyionS = np.zeros((npoints),float)
	energyionexc = np.zeros((npoints),float)
	energyexcion = np.zeros((npoints),float)
	importfile2 = np.loadtxt('table/tableEnergy.txt',dtype=3*'float,'+'float',delimiter='\t',usecols=list(range(4)),unpack=True)
	energyionS = np.array(importfile2[1]);energyionexc=np.array(importfile2[2]);energyexcion=np.array(importfile2[3])
	energyexcion = energyionexc/2.5e25
	#------------------------------------------------------------------------------
	return(mobilityInput,diffusionInput,energyionS,energyionexc,energyexcion)


#Interpolation FORMULA ======================================================================================
def Interpolation(fieldvalue,inputdat, interval, maximumvalue,error):
	#the transport, and reaction coefficients come from the table that has limited values. 
	#this function takes those values and uses linear approximation to fit the data for the
	#actual values that are obtained from the code
	xrow = inputdat.shape[0]
	try:
		ycol = inputdat.shape[1]
	except Exception as inst:
		ycol = xrow
		xrow = 1
	inputdata = np.zeros((xrow,ycol),float)
	inputdata[:,:] = inputdat.copy()
	fieldvalue[fieldvalue<-maximumvalue] = -(maximumvalue)
	fieldvalue[fieldvalue>maximumvalue] = maximumvalue
	fieldvalue = fieldvalue*interval
	indlocate = abs(fieldvalue[:]).astype(int)
	return(inputdata[:,indlocate]+(inputdata[:,indlocate+1]-inputdata[:,indlocate])*(abs(fieldvalue)-indlocate)/interval)
#--------------------------------------------------------------------------------------------------------------



def readParametersFromFile(param_name, filename):
    with open(filename, 'r') as f:
        for line in f:
            line = line.split("#")[0].strip() # remove comment after "#"
            if line:
                key, value = line.split('=')
                if key.strip() == param_name:
                    return value.strip()

    return None


# ------
def plotImage(title,storedensityy):
	plt.clf()
	plt.imshow(np.transpose(storedensityy),aspect = 'auto')
	plt.title(title)
	plt.colorbar()
	plt.savefig('output/' + title + '.png' , dpi=200)