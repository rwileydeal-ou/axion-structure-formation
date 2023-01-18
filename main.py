import dataclasses
from textwrap import dedent
import matplotlib as mpl
import numpy as np
from scipy.integrate import odeint
import scipy.special



# declare static global variables here
mPlanck = 2.4e18




################################################################
#                        Helper classes                        #
################################################################


@dataclasses.dataclass 
class InputData:
    mass_Modulus: float
    mass_WIMP: float
    crossSection_WIMP: float
    decayWidth_Modulus: float
    branchRatio_ModulusToWIMP: float
    fa: float
    temp_Reheat: float

@dataclasses.dataclass
class EnergyDensities:
    rho_Modulus: float
    rho_WIMP: float
    rho_Radiation: float
    n_Axion: float
    hubble: float
    rhoEQ_WIMP: float = 0.




################################################################
#               Primative helper functions                     #
################################################################


# this method computes axion mass based on current temperature (in GeV) and axion decay constant (in GeV)
def axionMass(fa, temp):
    m = np.power(0.078, 2.) / fa

    if (temp <= 0.2 ):
        return m
    elif ( temp > 0.2 and temp <= 1. ):
        return ( np.power(0.2 / temp, 6.5) * m )
    elif ( temp > 1. ):
        return ( np.power(0.2 / temp, 4.) * 0.018 * m )
    raise ValueError("No valid axion mass scenario!")

# this method computes dm_ax / dN utilizing a numerical approximation as described in notes
def axionMassDerivative( temp, fa, rho_Radiation, dRhoRad ):
    # if T < 0.2 GeV, mass is constant - no point in continuing 
    if (temp <= 0.2 ):
        return 0.

    # otherwise, first compute the dT/dN term
    dTdN = 0.25 * temp * ( dRhoRad / rho_Radiation )

    # now we can use the expression in the notes to compute prefactor of dm/dN
    dmdN = dTdN * np.power(0.078, 2.) / fa

    # finally, factor in temperature dependent cases
    if ( temp > 0.2 and temp <= 1. ):
        return ( - 6.5 / temp * np.power(0.2 / temp, 6.5) * dmdN )
    elif ( temp > 1. ):
        return ( - 4. / temp * np.power(0.2 / temp, 4.) * 0.018 * dmdN )
    raise ValueError("No valid axion mass scenario!")

# this method retrieves gstar from file for specified temperature
# if temperature does not match exact entry, interpolates between two nearest neighbors
def readGstarFromCSV( gstarCsvFile, temp ):
    return 225.

# this method computes the radiation temperature from the radiation energy density
# requires a CSV file containing gstar data, interpolates gstar with Temp estimations to find correct value
def temperature( rho_Radiation, gstarCsvFile ):
    gstarGuess = 100.

    temp0 = np.power( 30. * rho_Radiation / ( gstarGuess * np.power(np.pi, 2.) ), 0.25 )
    temp1 = 0.
    deltaTemp = 0.

    while ( np.abs( temp0 - temp1 ) / temp0 > 0.01 ):
        gstr = readGstarFromCSV( gstarCsvFile, temp0 )
        temp1 = np.power( 30. * rho_Radiation / ( gstr * np.power(np.pi, 2.) ), 0.25 )

        if ( np.abs( temp0 - temp1) == deltaTemp ):
            temp0 = ( temp0 + temp1 ) / 2.
        else:
            deltaTemp = np.abs( temp0 - temp1 )
            temp0 = temp1

    return temp1

# this method calculates rhoEquilibrium for WIMP
# see e.g. Eq.A.8 in arXiv: 1110.2491
def compute_rhoEquilibrium( temp, mass_WIMP ):
    g = 2

    if ( temp <= mass_WIMP / 10. ):
        # non-relativistic limit
        neq = g * np.power( mass_WIMP * temp / (2. * np.pi), 1.5 ) * np.exp( - mass_WIMP / temp )
        return neq * mass_WIMP
    elif ( temp < 1.5 * mass_WIMP and temp > mass_WIMP / 10. ):
        # intermediate regime
        neq = g * np.power(mass_WIMP, 2.) * temp * scipy.special.kn(2, mass_WIMP / temp) / ( 2. * np.power(np.pi, 2.) )
        f1 = mass_WIMP * scipy.special.kn(1, mass_WIMP / temp) / scipy.special.kn(2, mass_WIMP / temp)
        f2 = 3. * temp 
        return neq * ( f1 + f2 )
    elif ( temp >= 1.5 * mass_WIMP  ):
        # relativistic limit 
        return g * 7./8. * np.power(np.pi, 2.) / 30. * np.power( temp, 4. )
    raise ValueError("No valid temperature regime!")

def build_modulus_energyDensity_eqn( inputData, energyDensities ):
    decayWidth_Modulus = inputData.decayWidth_Modulus
    mass_Modulus = inputData.mass_Modulus
    hubble = energyDensities.hubble
    rho_Modulus = energyDensities.rho_Modulus

    # if 3H >~ m, modulus oscillations have not begun yet - return 0 since still frozen
    if ( mass_Modulus < 3. * hubble ):
        return 0.
    # hubble dilution
    dEdN = -3. * rho_Modulus
    # decay
    dEdN -= decayWidth_Modulus * rho_Modulus / hubble
    return dEdN

def build_WIMP_energyDensity_eqn( inputData, energyDensities ):
    crossSection_WIMP = inputData.crossSection_WIMP
    mass_WIMP = inputData.mass_WIMP
    decayWidth_Modulus = inputData.decayWidth_Modulus
    branchRatio_ModulusToWIMP = inputData.branchRatio_ModulusToWIMP
    hubble = energyDensities.hubble
    rho_WIMP = energyDensities.rho_WIMP
    rhoEQ_WIMP = energyDensities.rhoEQ_WIMP
    rho_Modulus = energyDensities.rho_Modulus

    # hubble dilution
    dEdN = -3. * rho_WIMP 
    # annihilations
    dEdN -= ( np.power(rho_WIMP, 2.) - np.power(rhoEQ_WIMP, 2.) ) * crossSection_WIMP / ( hubble * mass_WIMP ) 
    # injections
    dEdN += decayWidth_Modulus * rho_Modulus * branchRatio_ModulusToWIMP / hubble 
    return dEdN

def build_axion_numberDensity_eqn( mass_Axion, energyDensities ):
    # if 3H >~ m(T), axion oscillations have not begun yet - return 0 since still frozen
    if ( mass_Axion < 3. * energyDensities.hubble ):
        return 0.
    # hubble dilution 
    dndN = -3. * energyDensities.n_Axion
    return dndN

def build_radiation_energyDensity_eqn( inputData, energyDensities ):
    crossSection_WIMP = inputData.crossSection_WIMP
    mass_WIMP = inputData.mass_WIMP
    decayWidth_Modulus = inputData.decayWidth_Modulus
    branchRatio_ModulusToWIMP = inputData.branchRatio_ModulusToWIMP
    hubble = energyDensities.hubble
    rho_Radiation = energyDensities.rho_Radiation
    rho_WIMP = energyDensities.rho_WIMP
    rhoEQ_WIMP = energyDensities.rhoEQ_WIMP
    rho_Modulus = energyDensities.rho_Modulus

    # hubble dilution
    dEdN = -4. * rho_Radiation
    # annihilations 
    dEdN += ( np.power(rho_WIMP, 2.) - np.power(rhoEQ_WIMP, 2.) ) * crossSection_WIMP / ( mass_WIMP * hubble )
    # decays
    dEdN += decayWidth_Modulus * rho_Modulus * ( 1. - branchRatio_ModulusToWIMP ) / hubble
    return dEdN

def build_hubble_eqn( energyDensities, mass_Axion, dmAxdN ):
    dHdN = 0.
    dHdN -= ( 
        energyDensities.rho_Modulus 
        + energyDensities.rho_WIMP 
        + energyDensities.n_Axion * mass_Axion 
        + 4./3. * energyDensities.rho_Radiation 
        - 1./3. * dmAxdN * energyDensities.n_Axion 
    ) / ( 2. * energyDensities.hubble * np.power(mPlanck, 2.) )
    return dHdN




################################################################
#        Wrapped helper functions - Initial Conditions         #
################################################################

def compute_hubble_initialCondition( 
    inputData,
    gstar
):
    hubble0 = np.sqrt(
        np.power( np.pi, 2. ) * gstar / 90.
    ) * np.power( inputData.temp_Reheat, 2. ) / mPlanck
    return hubble0

def compute_radiation_initialCondition(
    inputData,
    gstar
):
    rho0_Rad = np.power( np.pi, 2. ) * gstar * np.power( inputData.temp_Reheat, 4. ) / 30.
    return rho0_Rad

def compute_modulus_initialCondition(
    inputData,
    gstar
):
    amplitude = mPlanck
    rho0_mod = 0.5 * np.power( inputData.mass_Modulus * amplitude, 2. )
    return rho0_mod

def compute_axion_initialCondition(
    inputData,
    gstar
):
    return 0.

def compute_zerothOrder_initialConditions(
    inputData,
    gstarCsvFile
):
    # read gstar once
    gstar = readGstarFromCSV( gstarCsvFile=gstarCsvFile, temp=inputData.temp_Reheat )

    hubble0 = compute_hubble_initialCondition( inputData=inputData, gstar=gstar )
    rho0_Radiation = compute_radiation_initialCondition( inputData=inputData, gstar=gstar )

    # for cases of interest, WIMP starts in equilibrium - just calculate rho_EQ at TR
    rho0_WIMP = compute_rhoEquilibrium( temp=inputData.temp_Reheat, mass_WIMP=inputData.mass_WIMP )

    rho0_Modulus = compute_modulus_initialCondition( inputData=inputData, gstar=gstar )
    n0_Axion = compute_axion_initialCondition( inputData=inputData, gstar=gstar )

    return [rho0_Modulus, rho0_WIMP, n0_Axion, rho0_Radiation, hubble0]

def compute_firstOrder_initialConditions():
    return[]


# this method computes the full set of initial conditions
def computeInitialConditions(
    inputData,
    gstarCsvFile
):
    zerothOrderY0 = compute_zerothOrder_initialConditions( inputData=inputData, gstarCsvFile=gstarCsvFile )

    # TODO: add in the first order eqns

    return zerothOrderY0




################################################################
#         Wrapped helper functions - Equation Builders         #
################################################################


# this helper method computes the zeroth order Boltzmann equations
def build_zerothOrder_equations( 
    inputData,
    energyDensities,
    mass_Axion, 
    temp
):
    dRhoPhi = build_modulus_energyDensity_eqn( inputData=inputData, energyDensities=energyDensities )
    dRhoChi = build_WIMP_energyDensity_eqn( inputData=inputData, energyDensities=energyDensities )
    dnAx = build_axion_numberDensity_eqn( mass_Axion, energyDensities=energyDensities )
    dRhoRad = build_radiation_energyDensity_eqn( inputData=inputData, energyDensities=energyDensities )

    # need dRho/dN to compute dmAx / dN (see notes)
    dmAxdN = axionMassDerivative(temp, inputData.fa, energyDensities.rho_Radiation, dRhoRad)

    dHubble = build_hubble_eqn( energyDensities=energyDensities, mass_Axion=mass_Axion, dmAxdN=dmAxdN )

    return[ dRhoPhi, dRhoChi, dnAx, dRhoRad, dHubble ]

# this helper method computes the first order Boltzmann equations
def build_firstOrder_equations():
    return[]

# this method computes the full set of Boltzmann equations
# "eqns" parameter is array containing numerical RHS of Boltz eqns
# "N" is number of e-folds
def build_Boltzmann_Equations( 
    eqns, 
    N, 
    inputData,
    gstarCsvFile
):
    if not dataclasses.is_dataclass( inputData ):
        raise TypeError("inputData must be of dataclass type InputData")

    # extract densities for modulus, WIMP, axion, and radiation, Hubble parameter, and ... first order ...
    # use axion number density instead of energy density since rho=n*m, but number density does not need numerical calculation of dm/dN
    rho_Modulus, rho_WIMP, n_Axion, rho_Radiation, hubble = eqns
    energyDensities = EnergyDensities( 
        rho_Modulus=rho_Modulus,
        rho_WIMP=rho_WIMP,
        rho_Radiation=rho_Radiation,
        n_Axion=n_Axion,
        hubble=hubble
     )

    # with the solution of previous iteration, need to compute required properties for this step
    # compute temperature 
    temp = temperature( rho_Radiation=rho_Radiation, gstarCsvFile=gstarCsvFile )
    # compute WIMP equilibrium energy density
    energyDensities.rhoEQ_WIMP = compute_rhoEquilibrium(temp, inputData.mass_WIMP)
    # compute axion mass
    mass_Axion = axionMass( fa=inputData.fa, temp=temp )

    if hubble == 0.:
        raise ValueError("Vanishing Hubble parameter!")

    zerothOrderEqns = build_zerothOrder_equations( 
        inputData,
        energyDensities,
        mass_Axion, 
        temp
    )
    
    # TODO: add in the first order eqns

    return zerothOrderEqns








################################################################
#                       Main functions                         #
################################################################



# this method builds and solves the Boltzmann equations
# initial conditions are assumed to be taken at inflationary reheating
def solveBoltzmannEquations(  ):
    # define number of e-folds
    N = np.linspace(0,20,100)

    gstarCsvFile = "mssm_gstar.csv"
    
    # NOTE: ALL DIMENSIONFUL VALUES SHOULD HAVE BASE UNITS OF GEV
    inputData = InputData( 
        mass_Modulus=5.e6, 
        mass_WIMP=200., 
        crossSection_WIMP=1.93e-08,
        decayWidth_Modulus = 1e-25,
        branchRatio_ModulusToWIMP = 0.2,
        fa = 1e11,
        temp_Reheat=1e12
    )

    # define initial conditions
    y0 = computeInitialConditions( 
        inputData=inputData,
        gstarCsvFile=gstarCsvFile
    )

    sol = odeint( 
        build_Boltzmann_Equations, 
        y0, 
        N, 
        args=( 
            inputData,
            gstarCsvFile
        ) 
    )

    print(sol)

solveBoltzmannEquations()
