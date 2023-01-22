from cmath import nan
from ctypes import util
from turtle import back
import matplotlib as mpl
import numpy as np
from scipy.integrate import ode
from scipy.integrate import Radau
from boltzmann import data
from boltzmann.eqns import util
from boltzmann.eqns import gstr
from boltzmann.eqns import zero
from boltzmann.eqns.initialconditions import zero as zeroIC

def compute_RelicDensity(
    rho, 
    temp, 
    gstarCsvFile
):
    rhoCr_0 = 8.0992e-47
    temp_0 = 2.34816e-13
    gstarEntropic_0 = 3.90909090909091

    gstarEntropic = gstr.readGstarFromCSV( gstarCsvFile=gstarCsvFile, temp=temp )

    oh2 = rho * ( gstarEntropic_0 / gstarEntropic ) * np.power( temp_0 / temp, 3. ) / rhoCr_0
    return oh2


################################################################
#         Wrapped helper functions - Equation Builders         #
################################################################

# this method computes the full set of initial conditions
def computeInitialConditions(
    inputData
):
    zerothOrderY0 = zeroIC.compute_zerothOrder_initialConditions( inputData=inputData )

    # TODO: add in the first order eqns

    return zerothOrderY0


# this method computes the full set of Boltzmann equations
# "eqns" parameter is array containing numerical RHS of Boltz eqns
# "N" is number of e-folds
def build_Boltzmann_Equations( 
    N, 
    eqns, 
    inputData
):
    if not data.dataclasses.is_dataclass( inputData ):
        raise TypeError("inputData must be of dataclass type InputData")

    # extract densities for modulus, WIMP, axion, and radiation, Hubble parameter, and ... first order ...
    # use axion number density instead of energy density since rho=n*m, but number density does not need numerical calculation of dm/dN
    rho_Modulus, rho_WIMP, n_Axion, rho_Radiation = eqns

    # check to see if modulus contributes to hubble constant
    modIsOsc = False
    hubble = 0.
    if rho_WIMP is not nan and rho_Radiation is not nan:
        hubble = np.sqrt( ( rho_Modulus + rho_WIMP + rho_Radiation ) / 3. ) / data.mPlanck

    # if modulus is oscillating, add it to hubble
    if inputData.mass_Modulus >= hubble and rho_Modulus > 0. and rho_Modulus is not nan:
        modIsOsc = True

    if rho_WIMP < 0.:
        raise ValueError("Bad WIMP convergence")

    energyDensities = data.EnergyDensities( 
        rho_Modulus=rho_Modulus,
        rho_WIMP=rho_WIMP,
        rho_Radiation=rho_Radiation,
        n_Axion=n_Axion,
        hubble=hubble,
        isOsc_Modulus=modIsOsc
     )

    # with the solution of previous iteration, need to compute required properties for this step
    # compute temperature 
    temp = util.temperature( rho_Radiation=rho_Radiation, gstarCsvFile=inputData.gstarCsvFile )

    # compute WIMP equilibrium energy density
    energyDensities.rhoEQ_WIMP = zero.compute_rhoEquilibrium(temp, inputData.mass_WIMP)
    # compute axion mass
    mass_Axion = util.axionMass( fa=inputData.fa, temp=temp )

    zerothOrderEqns = zero.build_zerothOrder_equations( 
        inputData,
        energyDensities,
        mass_Axion, 
        temp
    )
    
    # TODO: add in the first order eqns
    print(temp)
    print( rho_Modulus, rho_WIMP, energyDensities.rhoEQ_WIMP, n_Axion, rho_Radiation, hubble, zerothOrderEqns[0], zerothOrderEqns[1], zerothOrderEqns[2], zerothOrderEqns[3],)
    print("")
    return zerothOrderEqns



################################################################
#                       Main functions                         #
################################################################



# this method builds and solves the Boltzmann equations
# initial conditions are assumed to be taken at inflationary reheating
def solveBoltzmannEquations( 
    mass_Modulus, 
    mass_WIMP, 
    crossSection_WIMP,
    decayWidth_Modulus,
    branchRatio_ModulusToWIMP,
    fa,
    temp_Reheat,
    gstarCsvFile
):
    # define input data object first    
    inputData = data.InputData( 
        mass_Modulus = mass_Modulus, 
        mass_WIMP = mass_WIMP,
        crossSection_WIMP = crossSection_WIMP,
        decayWidth_Modulus = decayWidth_Modulus,
        branchRatio_ModulusToWIMP = branchRatio_ModulusToWIMP,
        fa = fa,
        temp_Reheat = temp_Reheat,
        gstarCsvFile = gstarCsvFile
    )

    # define initial conditions
    y0 = computeInitialConditions( 
        inputData=inputData
    )

    backend = "Radau"
    ode_solver = ode(
        build_Boltzmann_Equations
    ).set_integrator(
        backend,
        max_step = 0.01,
        rtol = 1e-8,
        atol = 1e-8,
        with_jacobian = True
    ).set_f_params(
        inputData
    )
    ode_solver.set_initial_value(y=y0, t=0.)

    t = 0.
    dt = 0.1

    while True:
        # t=0 already set in initial condition, increment first
        t += dt
        y = ode_solver.integrate( t )
        print(y)
        ode_solver.set_initial_value( y, t )
        # evolve until modulus is decayed
        if float(y[0]) < 1e-65:
            break

    temp = zero.temperature( 
        float(y[3]), 
        gstarCsvFile = inputData.gstarCsvFile 
    )
    oh2 = compute_RelicDensity( 
        float(y[1]), 
        temp=temp, 
        gstarCsvFile = inputData.gstarCsvFile 
    )
    print(oh2)
