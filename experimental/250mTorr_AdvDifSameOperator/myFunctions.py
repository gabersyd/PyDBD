import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.sparse.linalg as la
import scipy.sparse as sparse
import time as tm
import logging


####==============   POISSON MATRIX USED TO SOLVE THE POISSON'S EQUATION IMPLICTLY   ========================
#============================================================================================================
def SparseLaplacianOperator(n,k1=-1,k2=0,k3=1):
	#create tridiagonal (n x n) matrix with -2 on the main diagonal and 1 above and below the main diag.
	d1 = np.zeros((n),float)
	d2 = np.ones((n),float)
	d3 = np.zeros((n),float)
	d1[:-2] = 1; d2[1:-1]=-2;d3[2:]=1
	return (sparse.dia_matrix(([d1,d2,d3],[k1,k2,k3]),shape=(n,n)).tocsc() ) 
#------------------------------------------------------------------------------------------------------------
	
	
def driftDiffusionExplicitOperator(ngrid0,inputdataa,diffusiondata,dx,dt,velocity): # this needs to be either documented or simplified later or both
	# advection ---- 
	flux = (0.5*(velocity[1:]*inputdataa[1:]+velocity[:-1]*inputdataa[:-1])-
		  0.5*0.5*abs(velocity[1:]+velocity[:-1])*(inputdataa[1:]-inputdataa[:-1]))*dt
	inputdataa[1:-1] += -(flux[1:]-flux[:-1])/dx
	# diffusion ----
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


def importtransportdiffusion():
	#------ Importing mobility and diffusion as a function of efield---------------
	#==============================================================================
	parameterSize = 999 # change this to get read dynamic value
	importfile = np.loadtxt('table/tableEfield.txt',dtype=6*'float,'+'float',delimiter='\t',usecols=list(range(7)),skiprows=1,unpack=True)
	mobilityInput = np.zeros((4,parameterSize),float)
	diffusionInput = np.zeros((4,parameterSize),float)
	mobilityInput[0,:] = np.array(importfile[0]);mobilityInput[1,:]=np.array(importfile[1]);mobilityInput[2,:]=np.array(importfile[2])
	diffusionInput[0,:] = np.array(importfile[3]);diffusionInput[1,:]=np.array(importfile[4]);diffusionInput[2,:]=np.array(importfile[5]);diffusionInput[3,:]=np.array(importfile[6])
	#------------ Importing reaction rates as a function of efield-----------------
	#==============================================================================
	npoints = 190 # change this to get dynamic value
	energyionS = np.zeros((npoints),float)
	energyionexc = np.zeros((npoints),float)
	energyexcion = np.zeros((npoints),float)
	importfile2 = np.loadtxt('table/tableEnergy.txt',dtype=3*'float,'+'float',delimiter='\t',usecols=list(range(4)),unpack=True)
	energyionS = np.array(importfile2[1]);energyionexc=np.array(importfile2[2]);energyexcion=np.array(importfile2[3])
	energyexcion = energyionexc/2.5e25
	#------------------------------------------------------------------------------
	return(mobilityInput,diffusionInput,energyionS,energyionexc,energyexcion)


#Interpolation FORMULA ======================================================================================
def Interpolation(fieldvalue,inputdat, interval, maximumvalue,error): # redo this directly use python interpolation func
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


# -- reading input parameters from file ---------
def readParametersFromFile(param_name, filename):
    with open(filename, 'r') as f:
        for line in f:
            line = line.split("#")[0].strip() # remove comment after "#"
            if line:
                key, value = line.split('=')
                if key.strip() == param_name:
                    return value.strip()

    return None


# ------ plotting and saving results ---- 
def plotImageAndSaveResult(title,twoDMatrix):
	plt.clf()
	plt.imshow(np.transpose(twoDMatrix),aspect = 'auto')
	plt.title(title)
	plt.colorbar()
	plt.savefig('output/' + title + '.png' , dpi=200)
	np.savetxt('output/' + title + '.txt',twoDMatrix)