# ___________________________________________________________________________________________
#______________	 PYTHON CODE ON 1D ARGON PLASMA   (04-09-2023)  _____________________________
# to cite the code please cite :: https://iopscience.iop.org/article/10.1088/2058-6272/ac241f
#_____________________________________________________________________________________________
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.sparse.linalg as la
import scipy.sparse as sparse
import time as tm
import logging
import myFunctions

# parameterSize needs to be removed later
parameterSize=996						# NUMBER OF ROWS IN THE INPUT TEXT FILE


# -------------------------- READING CONDITIONS FROM INPUT FILE ------------------------------------------------
ngrid0 		= (int(myFunctions.readParametersFromFile("number_of_grids", "conditions.txt")))
gasWidth 	= (float(myFunctions.readParametersFromFile("gas_width", "conditions.txt")))
pressure	= (float(myFunctions.readParametersFromFile("gas_pressure", "conditions.txt")))
temperature = (float(myFunctions.readParametersFromFile("gas_temperature", "conditions.txt")))
gamma		= (float(myFunctions.readParametersFromFile("secondary_electron_emission_coefficient", "conditions.txt")))
volt		= (float(myFunctions.readParametersFromFile("voltage", "conditions.txt")))
frequencySource		= (float(myFunctions.readParametersFromFile("frequency", "conditions.txt")))
initialNumberDensity = float(myFunctions.readParametersFromFile("initial_number_density", "conditions.txt"))
dt = float(myFunctions.readParametersFromFile("timeStep", "conditions.txt"))
totalcycles = int(myFunctions.readParametersFromFile("total_AC_simulation_cycles", "conditions.txt"))
gasConstant = float(myFunctions.readParametersFromFile("gas_constant", "conditions.txt"))
useAdaptiveTime = bool((myFunctions.readParametersFromFile("enable_adaptive_time_stepping", "conditions.txt"))) # feature not used here
secondaryElectronEmission = float(myFunctions.readParametersFromFile("secondary_electron_emission", "conditions.txt"))
sectemp = float(myFunctions.readParametersFromFile("secondary_electron_temperature", "conditions.txt"))
seedElectrons = float(myFunctions.readParametersFromFile("seed_electron_density", "conditions.txt"))
maxeTemp = float(myFunctions.readParametersFromFile("electron_temp_max", "conditions.txt"))
mineTemp = float(myFunctions.readParametersFromFile("electron_temp_min", "conditions.txt"))

# Printing Conditions. remove later - - 
print (ngrid0, gasWidth, pressure, temperature, gamma, volt, frequencySource, initialNumberDensity, dt, totalcycles)



#---------------------------------- CONSTANTS ----------------------------------------------------------------
avogadro = 6.02e23 						# molecules per mole
ev = 1.6e-19								# conversion unit ev to joule
ee = 1.6e-19							# electronic charge
e0 = 8.54187817e-12						# permittivity of free space
Kboltz = 1.380e-23						# Boltzmann constant

#---------------------------------- CALCULATIONS --------------------------------------------------------------
dx = gasWidth * 10 ** (-3) / (ngrid0+1.0)		# Grid size in meter
inelec = gasWidth * 10 ** (-3)		# total interelectrode separation
ngrid = int(ngrid0 + 2)			# total number of grid points(2 dielectrics +gas medium + all edge points)
gasdens = (pressure * avogadro) / (gasConstant * temperature)  # ideal gas law
townsendunit = 1.0 / (gasdens * 1e-21)	# townsend factor to convert from V/m to townsends unit

# to be removed after implementing arbitrary chemistry ---------------------
ns = 4										# total number of species
nr = 5										# total number of chemical reactions

#*** Initialization
#-----------------------------------------------------------------------------------------------------
ndensity = np.zeros((ns, ngrid0 + 2), float)	# number density of each species
ncharge = np.array([-1, 1, 1, 0])				# corresponding charge of the each species
netcharge = np.zeros(ngrid, float)			# net charge at each grid points
potentl = np.zeros(ngrid, float)				# potential at each grid points
efield = np.zeros(ngrid0 + 2, float)			# electric field at each grid points
efieldTd = np.zeros(ngrid0 + 2, float)			# electric field at each grid points
efieldPP = np.zeros(ngrid0 + 2, float)			# electric field at each grid points

mobilityG = np.zeros((ns , ngrid0 + 2), float)	# mobility at each grid points
diffusionG = np.zeros((ns , ngrid0 + 2), float)	# diffusion coefficient at grid points
sourceG = np.zeros((nr, ngrid0 + 2), float)		# source at each grid points
react = np.zeros((4, ngrid0 + 2), float)		# rate of production of each plasma species 
R = np.zeros((5, ngrid0 + 2), float)			# reaction rate for individual reactions considered in this model 
sigmaLR = np.zeros((ns, 2), float)			# surface charge density
esigmaLR = np.zeros((2),float)				# surface energy density
ndensity = initialNumberDensity + 0 * np.random.rand(ns, ngrid0 + 2)		# initializing the number densities with random value
ndensity[2] = 0.							# initially argon dimer density is considered zero
edensity = 0 * np.random.rand(ngrid0 + 2)		# energy density initialization with zero value
datapercycle = 25							# how many data per cycle to record
datapercycle1 = 25							# how many data per cycle to record

totaldata = totalcycles * datapercycle		# calculation of how many data will there be in total
totaltime = totalcycles / frequencySource		# calculating the total simulation time
stepinterval = totaltime / totaldata			# calculating approdimate time between two saving points

totaldata1 = totalcycles * datapercycle1		# calculation of how many data will there be in total
stepinterval1 = totaltime / totaldata1		# calculating approdimate time between two saving points

prevloc = 0									# accumulator (that will be used to take decision to save data)
prevloc1 = 0								# accumulator (that will be used to take decision to save data)

storedensity = np.zeros((totaldata + 1, ns, ngrid0 + 2), float)	# number density	
storenetcharge = np.zeros((totaldata + 1, ngrid0 + 2), float)		# net charge
storeefield = np.zeros((totaldata + 1, ngrid0 + 2), float)		# elecritc field
storepotentl = np.zeros((totaldata + 1, ngrid0 + 2), float)		# potential
storeenergy = np.zeros((totaldata + 1, ngrid0 + 2), float)					# potential
storeReact = np.zeros((totaldata + 1, ns,ngrid0 + 2), float)					# production rate
storeR = np.zeros((totaldata + 1, nr, ngrid0 + 2), float)						# reaction rate
storeCurrent = np.zeros(int(totaldata1 + 1), float)						# current
storetime = np.zeros(int(totaldata1 + 1), float)						# current

(mobilityInput, diffusionInput, energyionS, energyionexc, energyexcion) = myFunctions.importtransportdiffusion()
poissonSparseMatrix = myFunctions.SparseLaplacianOperator(ngrid)   #poisson equation solving matrix


#==================================================================================================
#									 *** TIME LOOP ***
starttime = tm.time()
time = 0

trT1 = 0.0
trT2 = 0.0
trT3 = 0.0
lpT = 0.0 

try:
	while time < totaltime: # and elapsed<0.1 :
		inst1 = tm.time()
		time = time + dt
		newloc = int(time / stepinterval)
		if newloc > prevloc:
			save = 1
			prevloc = newloc
		else:
			save = 0
			newloc1 = int(time / stepinterval1)
		if newloc1 > prevloc1:
			save1 = 1
			prevloc1 = newloc1
		else:
			save1 = 0
		#-----------------------------------------------------------------------------------------------
		#							   *** Energy Source ***
		juoleheating = (efield[1:-1] * mobilityG[0, 1:-1] * ndensity[0, 1:-1] - diffusionG[0, 1:-1] * ((ndensity[0, 2:] - ndensity[0, :-2]) / (2 * dx)))  # Juole heating term (energy source)
		energySource =- ee * juoleheating * efield[1:-1]-1 * (15.80 * ev * R[0, 1:-1] + 11.50 * ev * R[1, 1:-1] - 15.80 * ev * R[3, 1:-1] + 4.43 * ev * R[4, 1:-1]) / dt
		edensity[1:-1] = edensity[1:-1] + dt * energySource


		#========================================================================================================
		#							   *** PARTICLE SOURCE TERM ***
		backgroundradiation = 0 * 1e4												# Small number; for ionization due to cosmic radiation
		R[0] = sourceG[0] * (ndensity[0] + backgroundradiation) * gasdens * dt			# source for reaction Ar + e- = Ar+ + 2e-
		R[1] = 1 * (ndensity[0] + backgroundradiation) * gasdens * sourceG[1] * dt		# source for reaction Ar + e- = Ar* + 2e-
		R[2] = 2.5e-43 * gasdens * gasdens * ndensity[1] * dt							# source for reaction Ar + Ar +Ar+ = Ar_2^+ + 2e-
		R[3] = 3e-14 * (ndensity[0] + backgroundradiation) * ndensity[2] * dt			# source for reaction Ar + e- = Ar+ + 2e-
		R[4] = 1 * sourceG[4] * (ndensity[0] + backgroundradiation) * ndensity[3] * dt	# source for reaction Ar* + e- = Ar + e-
		#--------------------------------------------------------------------------------------------------------
		react[0] = (R[0] + R[4] - R[3])				# production of particle [0]
		react[1] = (R[0] + R[4] - R[2])				# production of particle [1]
		react[2] = (R[2] - R[3])					# production of particle [2]
		react[3] = ( -R[4] + R[1])					# production of particle [3]
		ndensity[:,1:-1] += 1 * react[:, 1:-1]		# adding newly produced particles to the gas

		#======================================================================================================
		#								   *** Particle and Energy Transport ***
		for loopDD in np.arange(ns):
			mobilityG[loopDD] = ncharge[loopDD] * np.interp( np.abs(efieldTd), np.arange(990),  mobilityInput[loopDD, :990] ) / gasdens	
			diffusionG[loopDD] = np.interp( np.abs(efieldTd), np.arange(990),  diffusionInput[loopDD, :990] ) / gasdens	
			ndensity[loopDD, 1:-1] = myFunctions.driftDiffusionExplicitOperator(ngrid0, ndensity[loopDD,:],diffusionG[loopDD,:], dx, dt, mobilityG[loopDD,] * efield)

		edensity[1:-1] = myFunctions.driftDiffusionExplicitOperator(ngrid0, edensity, (5/3) * diffusionG[0], dx, dt, (5/3) * mobilityG[0] * efield)

		inst2 = tm.time()
		#==================================================================================================
		#						   *** POISSON'S EQUATION ***
		netcharge = ee * np.dot(ncharge,ndensity)					# calculating net charge
		leftPot = 1.0 * volt * np.sin( 2 * np.pi * time * frequencySource)	   					# applied voltage (left)
		rightpot = 0.0														# ground
		chrgg =- (netcharge / e0) * dx * dx							 		# RHS matrix. <Read documentation>
		chrgg[0] = leftPot													# left boundary condition
		chrgg[-1] = rightpot										  		# right boundary condition
		potentl = la.spsolve(poissonSparseMatrix, chrgg)			   			# solving system of Matrix equations

		#electric field 
		efield[1:-1] =   - (potentl[2:] - potentl[:-2]) / (2.0 * dx)
		efield[0] =   - (potentl[1] - potentl[0]) / dx
		efield[-1] =   - (potentl[-1] - potentl[-2]) / dx
		efieldTd = townsendunit * efield[:]
		#----------------------------------------------------------------------------------------------------------
		inst3 = tm.time()

		#==================================================================================================
		#					  *** TRANSPORT AND REACTION COEFFICIENTS ***
		#--------------------------------------------------------------------------------------------------
		energyparticle = edensity / (ndensity[0] + 1e-4) / ev
		energyparticle = np.clip(energyparticle, 0, 16.99)
		sourceG[0] = np.interp(energyparticle, np.arange(200) / 10,  energyionS) 	# reaction rate
		sourceG[1] = np.interp(energyparticle, np.arange(200) / 10,  energyionexc) 	# reaction rate
		sourceG[2] = np.interp(energyparticle, np.arange(200) / 10,  energyexcion) 	# reaction rate
		#------------------------------------------------------------------------------------------------
		inst4 = tm.time()

		#========================================================================================================
		#				   *** BOUNDARY CONDITION (THERMAL VELICITY) ***
		eTemp = (2/3) * edensity / (ndensity[0] + 1e-4) / Kboltz
		eTemp = np.clip(eTemp, mineTemp, maxeTemp)
		vthermal = (1/2) * (8 * Kboltz * eTemp/(3.14*9.11e-31)) ** (1/2)

		ndensity[0,1] = (ndensity[0,1] * dx + dt * (-ndensity[0,1] * vthermal[1])) / dx
		ndensity[0,-2] = (ndensity[0,-2] * dx + dt * (-ndensity[0,-2] * vthermal[1])) / dx
		edensity[1] = (edensity[1] * dx + dt * (-(5 / 3) * edensity[1] * vthermal[-2])) / dx
		edensity[-2] = (edensity[-2] * dx + dt * (-(5 / 3) * edensity[-2] * vthermal[-2])) / dx
		
		# seed electron contribution (floor value) -------
		temmatrix = seedElectrons + 0 * ndensity[0].copy()
		temmatrix[ndensity[0] > seedElectrons] = 0.0	 
		ndensity[0,1:-1] += temmatrix[1:-1]
		ndensity[1,1:-1] += temmatrix[1:-1]	

		#===============================================================================================
		#							*** CURRENT CALCULATION ***
		current = (ee * np.sum((efield[2:-2] * mobilityG[1,2:-2] * ndensity[1, 2:-2]
				+  1 * efield[2:-2] * mobilityG[2,2:-2] * ndensity[2, 2:-2] +
				efield[2:-2] * mobilityG[0,2:-2] * ndensity[0,2:-2] - 
				1 * diffusionG[2,2:-2] * (ndensity[2,3:-1] -ndensity[2, 1:-3]) / (2 * dx)
				- 1 * diffusionG[1,2:-2] * (ndensity[1,3:-1] - ndensity[1, 1:-3]) / (2 * dx)
				+ 1 * diffusionG[0,2:-2] * (ndensity[0,3:-1] - ndensity[0, 1:-3]) / (2 * dx)) * dx)
				+1 * e0 * np.sum(efieldPP[5:-5] - efield[5:-5]) * dx)

		#======================================STORING RESULTS==========================================
		if (save == 1):
			storedensity[newloc,:,:] = ndensity[:,:]
			storenetcharge[newloc] = netcharge
			storeefield[newloc] = efield
			storepotentl[newloc] = potentl
			storeenergy[newloc] = energyparticle
			storeR[newloc] = R
			storetime[newloc] = time
		#-----------------------------------------------------------------------------------------------
		if (save1 == 1):
			storeCurrent[newloc1] = current
		#-----------------------------------------------------------------------------------------------
		currenttime = tm.time()
		inst5 = tm.time(); trT1 += (inst2 - inst1); trT2 += (inst4 - inst3); trT3 += (inst5 - inst4); lpT +=  (inst3 - inst2) 
		elapsed = (currenttime - starttime) / 3600
except Exception as e: 
	print(e)
	rank = 1
	# TODO: add some code that will save all intermediate results
	np.savetxt('output/parameters' + str(rank) + '.txt', np.array([newloc, ngrid0, ngrid, elapsed]))
	np.savetxt('output/error.txt', str(e))

print("total elasped", elapsed * 3600)
print("Percentage time for transport and source term", trT1 / (trT1 + trT2 + trT3 + lpT) * 100)
print("Percentage time for interpolation", trT2 / (trT1 + trT2 + trT3 + lpT) * 100)
print("Percentage time for boundary condition and storing results", trT3 / (trT1 + trT2 + trT3 + lpT) * 100)
print("Percentage time for laplacian only", lpT / (trT1 + trT2 + trT3 + lpT) * 100)
# ----------------------------- Save Results ------------------------------
# -------------------------------------------------------------------------
speciesList = np.array(['electron', 'Ar+', 'Ar2+', 'Ar*'])
reactionList = np.array(['ionization',
			 'excitation',
			 'Dimer_formation',
			 'R4', 
			 'metastable_Ionization'])
storedensity[storedensity < 0] = seedElectrons
for ck in np.arange(ns):
	myFunctions.plotImageAndSaveResult( dx * np.arange(ngrid), storetime, speciesList[ck] + ' density$(1/m^3)$', storedensity[:,ck,:] )
	myFunctions.plotImageAndSaveResult( dx * np.arange(ngrid), storetime, 'production ' + speciesList[ck]+ '${1/m3}$', storeReact[:,ck,:] )
myFunctions.plotImageAndSaveResult( dx * np.arange(ngrid), storetime, 'Potential $(V)$', storepotentl )
myFunctions.plotImageAndSaveResult( dx * np.arange(ngrid), storetime, 'net Charge $(C)$', storenetcharge )
myFunctions.plotImageAndSaveResult( dx * np.arange(ngrid), storetime, 'efield $(V/m)$', storeefield )
myFunctions.plotImageAndSaveResult( dx * np.arange(ngrid), storetime, 'energy $(eV)$', storeenergy )

for ck in np.arange(nr):
	myFunctions.plotImageAndSaveResult( dx * np.arange(ngrid) , storetime, reactionList[ck] + 'Rate $(1/m^3)$', storeR[:,ck,:] )
np.savetxt( 'output/current.txt', storeCurrent )
np.savetxt( 'output/parameters.txt',np.array([newloc, ngrid0, ngrid, elapsed]) )
