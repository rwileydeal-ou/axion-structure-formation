from cmath import nan
from ctypes import util
from turtle import back
import matplotlib as mpl
import numpy as np
import csv
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
    # extract densities for modulus, WIMP, axion, and radiation, and ... first order ...
    # use axion number density instead of energy density since rho=n*m, but number density does not need numerical calculation of dm/dN
    rho_Modulus, rho_WIMP, n_Axion, rho_Radiation = eqns

    # with the solution of previous iteration, need to compute required properties for this step
    # compute temperature 
    temp = util.temperature( rho_Radiation=rho_Radiation, gstarCsvFile=inputData.gstarCsvFile )

    # compute WIMP equilibrium energy density
    rhoEQ_WIMP = zero.compute_rhoEquilibrium(temp, inputData.mass_WIMP)
    # compute axion mass
    mass_Axion = util.axionMass( fa=inputData.fa, temp=temp )

    hubble = np.sqrt( 
        ( rho_Modulus + rho_WIMP + rho_Radiation + n_Axion * mass_Axion ) / 3. 
    ) / data.mPlanck

    energyDensities = data.EnergyDensities( 
        rho_Modulus=rho_Modulus,
        rho_WIMP=rho_WIMP,
        rho_Radiation=rho_Radiation,
        n_Axion=n_Axion,
        rhoEQ_WIMP=rhoEQ_WIMP,
        hubble=hubble
    )

    zerothOrderEqns = zero.build_zerothOrder_equations( 
        inputData,
        energyDensities,
        mass_Axion, 
        temp
    )
    
    # TODO: add in the first order eqns

    return zerothOrderEqns[0]












def build_Boltzmann_Jacobian( 
    N, 
    eqns, 
    inputData
):
    # extract densities for modulus, WIMP, axion, and radiation, Hubble parameter, and ... first order ...
    # use axion number density instead of energy density since rho=n*m, but number density does not need numerical calculation of dm/dN
    rho_Modulus, rho_WIMP, n_Axion, rho_Radiation = eqns

    # with the solution of previous iteration, need to compute required properties for this step
    # compute temperature 
    temp = util.temperature( rho_Radiation=rho_Radiation, gstarCsvFile=inputData.gstarCsvFile )

    # compute WIMP equilibrium energy density
    rhoEQ_WIMP = zero.compute_rhoEquilibrium(temp, inputData.mass_WIMP)
    # compute axion mass
    mass_Axion = util.axionMass( fa=inputData.fa, temp=temp )

    hubble = np.sqrt( 
        ( rho_Modulus + rho_WIMP + rho_Radiation + n_Axion * mass_Axion ) / 3. 
    ) / data.mPlanck

    energyDensities = data.EnergyDensities( 
        rho_Modulus=rho_Modulus,
        rho_WIMP=rho_WIMP,
        rho_Radiation=rho_Radiation,
        n_Axion=n_Axion,
        rhoEQ_WIMP=rhoEQ_WIMP,
        hubble=hubble
    )

    zerothOrderEqns = zero.build_zerothOrder_equations( 
        inputData,
        energyDensities,
        mass_Axion, 
        temp
    )
    
    return zerothOrderEqns[1]



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
    thetaI,
    temp_Reheat,
    gstarCsvFile,
    outputCsv
):
    # define input data object first    
    inputData = data.InputData( 
        mass_Modulus = mass_Modulus, 
        mass_WIMP = mass_WIMP,
        crossSection_WIMP = crossSection_WIMP,
        decayWidth_Modulus = decayWidth_Modulus,
        branchRatio_ModulusToWIMP = branchRatio_ModulusToWIMP,
        fa = fa,
        thetaI = thetaI,
        temp_Reheat = temp_Reheat,
        gstarCsvFile = gstarCsvFile
    )

    # define initial conditions
    y0 = computeInitialConditions( 
        inputData=inputData
    )

    kwargs = dict(method='bdf')
    ode_solver = ode(
        build_Boltzmann_Equations,
        jac=build_Boltzmann_Jacobian
    ).set_integrator(
        "lsoda",
        nsteps=10000,
        atol=1e-12,
        rtol=1e-12,
        max_step=0.000001,
        **kwargs
    ).set_f_params(
        inputData
    ).set_jac_params(
        inputData
    )
    ode_solver.set_initial_value(y=y0, t=0.)

    dt = 0.1

    while True:
        # t=0 already set in initial condition, increment first
        y = ode_solver.integrate( ode_solver.t + dt )
        print(y)

        tempTest = util.temperature( 
            float(y[3]), 
            gstarCsvFile = inputData.gstarCsvFile 
        )

        mass_Axion = util.axionMass( inputData.fa, tempTest )
        hubble = np.sqrt( 
            ( y[0] + y[1] + y[3] + y[2] * mass_Axion ) / 3. 
        ) / data.mPlanck

        # update axion initial condition since m varies
        if hubble >= mass_Axion:
            y[2] = zeroIC.compute_axion_initialCondition( inputData=inputData, mass_Axion=mass_Axion )

        # now update full set of ICs
        ode_solver.set_initial_value( y, ode_solver.t )
        # evolve until modulus is decayed
        if float(y[0]) < 1e-85 and tempTest < 1e-4:
            break

    temp = util.temperature( 
        float(y[3]), 
        gstarCsvFile = inputData.gstarCsvFile 
    )
    oh2_WIMP = compute_RelicDensity( 
        float(y[1]), 
        temp=temp, 
        gstarCsvFile = inputData.gstarCsvFile 
    )
    oh2_Axion = compute_RelicDensity(
        float(y[2]) * util.axionMass( inputData.fa, temp=temp ),
        temp=temp,
        gstarCsvFile = inputData.gstarCsvFile
    )

    with open(outputCsv, 'a', newline='') as f:
        writer = csv.writer(f) 
        writer.writerow( [ mass_Modulus, mass_WIMP, crossSection_WIMP, oh2_WIMP, oh2_Axion ] ) 

