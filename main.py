from textwrap import dedent
import numpy as np
from scipy.integrate import odeint

# declare static global variables here
mPlanck = 2.4e18

################################################################
#               Primative helper functions                     #
################################################################

# this method computes axion mass based on current temperature (in GeV) and axion decay constant (in GeV)
def axionMass(fa, temp):
    m = ( 0.078 )**2. / fa

    if (temp <= 0.2 ):
        return m
    elif ( temp > 0.2 and temp <= 1. ):
        return ( (0.2 / temp)**6.5 * m )
    elif ( temp > 1. ):
        return ( (0.2 / temp)**4. * 0.018 * m )

    raise ValueError("No valid axion mass scenario!")

# this method computes dm_ax / dN utilizing a numerical approximation as described in notes
def axionMassDerivative():
    return 0.

# this method retrieves gstar from file for specified temperature
# if temperature does not match exact entry, interpolates between two nearest neighbors
def readGstarFromCSV( gstarCsvFile, temp ):
    return 0.

# this method computes the radiation temperature from the radiation energy density
# requires a CSV file containing gstar data, interpolates gstar with Temp estimations to find correct value
def temperature( rhoRadiation, gstarCsvFile ):
    gstarGuess = 100.

    temp0 = ( 30. * rhoRadiation / ( gstarGuess * np.Pi**2. ))**(1./4.)
    temp1 = 0.
    deltaTemp = 0.

    while ( np.abs( temp0 - temp1 ) / temp0 > 0.01 ):
        gstr = readGstarFromCSV( gstarCsvFile, temp0 )
        temp1 = ( 30. * rhoRadiation / ( gstr * np.Pi**2. ))**(1./4.)

        if ( np.abs( temp0 - temp1) == deltaTemp ):
            temp0 = ( temp0 + temp1 ) / 2.
        else:
            deltaTemp = np.abs( temp0 - temp1 )
            temp0 = temp1

    return temp1

# this method calculates nEquilibrium for WIMP
def compute_nEquilibrium():
    nEq = 0.

    return nEq

def build_modulus_energyDensity_eqn( rhoPhi, H, GammaPhi, mPhi ):
    # if 3H >~ m, modulus oscillations have not begun yet - return 0 since still frozen
    if ( mPhi < 3. * H ):
        return 0.
    # hubble dilution
    dEdN = -3. * rhoPhi
    # decay
    dEdN -= GammaPhi * rhoPhi / H
    return dEdN

def build_WIMP_energyDensity_eqn( rhoPhi, rhoChi, H, rhoEquilChi, sigVChi, mChi, GammaPhi, BrPhiWIMP ):
    # hubble dilution
    dEdN = -3. * rhoChi 
    # annihilations
    dEdN -= ( rhoChi**2. - rhoEquilChi**2. ) * sigVChi / ( H * mChi ) 
    # injections
    dEdN += GammaPhi * rhoPhi * BrPhiWIMP / H 
    return dEdN

def build_axion_numberDensity_eqn( nAxion, mAxion, H ):
    # if 3H >~ m(T), axion oscillations have not begun yet - return 0 since still frozen
    if ( mAxion < 3. * H ):
        return 0.
    # hubble dilution 
    dndN = -3. * nAxion
    return dndN

def build_radiation_energyDensity_eqn( rhoPhi, rhoChi, rhoRad, H, rhoEquilChi, sigVChi, mChi, GammaPhi, BrPhiWIMP ):
    # hubble dilution
    dEdN = -4. * rhoRad
    # annihilations 
    dEdN += ( rhoChi**2. - rhoEquilChi**2. ) * sigVChi / ( mChi * H )
    # decays
    dEdN += GammaPhi * rhoPhi * ( 1. - BrPhiWIMP ) / H
    return dEdN

def build_hubble_eqn( rhoPhi, rhoChi, nAx, rhoRad, H, mAxion, dmAxdN ):
    dHdN = 0.
    dHdN -= ( rhoPhi + rhoChi + nAx * mAxion + 4./3. * rhoRad - 1./3. * dmAxdN * nAx ) / ( 2. * H * mPlanck**2. )
    return dHdN



################################################################
#                 Wrapped helper functions                     #
################################################################


def compute_zerothOrder_initialConditions():
    return[]


# this helper method computes the zeroth order Boltzmann equations
def build_zerothOrder_equations( rhoPhi, rhoChi, nAx, rhoRad, H, sigVChi, mChi, GammaPhi, BrPhiWIMP, mPhi, mAxion, dmAxdN ):
    
    rhoEquilChi = 0.

    dRhoPhi = build_modulus_energyDensity_eqn( rhoPhi, H, GammaPhi, mPhi )
    dRhoChi = build_WIMP_energyDensity_eqn( rhoPhi, rhoChi, H, rhoEquilChi, sigVChi, mChi, GammaPhi, BrPhiWIMP )
    dnAx = build_axion_numberDensity_eqn( nAx, mAxion, H )
    dRhoRad = build_radiation_energyDensity_eqn( rhoPhi, rhoChi, rhoRad, H, rhoEquilChi, sigVChi, mChi, GammaPhi, BrPhiWIMP )
    dHubble = build_hubble_eqn( rhoPhi, rhoChi, nAx, rhoRad, H, mAxion, dmAxdN )

    return[ dRhoPhi, dRhoChi, dnAx, dRhoRad, dHubble ]

def compute_firstOrder_initialConditions():
    return[]

# this helper method computes the first order Boltzmann equations
def build_firstOrder_equations():
    return[]


# this method computes the full set of Boltzmann equations
# "eqns" parameter is array containing numerical RHS of Boltz eqns
# "N" is number of e-folds
def build_Boltzmann_Equations( eqns, N, sigVChi, mChi, GammaPhi, BrPhiWIMP, gstarCsvFile, fa, mPhi ):
    # extract densities for modulus, WIMP, axion, and radiation, Hubble parameter, and ... first order ...
    # use axion number density instead of energy density since rho=n*m, but number density does not need numerical calculation of dm/dN
    rhoPhi, rhoChi, nAx, rhoRad, H = eqns

    # with the solution of previous iteration, need to compute required properties for this step
    # compute temperature 
    temp = temperature( rhoRadiation=rhoRad, gstarCsvFile=gstarCsvFile )
    # compute axion mass
    mAxion = axionMass( fa=fa, temp=temp )
    # estimate dm_ax / dN
    dmAxdN = axionMassDerivative()

    if H == 0.:
        raise ValueError("Vanishing Hubble parameter!")

    zerothOrderEqns = build_zerothOrder_equations( rhoPhi, rhoChi, nAx, rhoRad, H, sigVChi, mChi, GammaPhi, BrPhiWIMP, mPhi, mAxion, dmAxdN )
    
    # TODO: add in the first order eqns

    return zerothOrderEqns

# this method computes the full set of initial conditions
def computeInitialConditions():
    zerothOrderY0 = compute_zerothOrder_initialConditions()

    # TODO: add in the first order eqns

    return zerothOrderY0











################################################################
#                       Main functions                         #
################################################################



# this method builds and solves the Boltzmann equations
# initial conditions are assumed to be taken at inflationary reheating
def solveBoltzmannEquations(  ):
    # define number of e-folds
    N = np.linspace(0,20,100)

    # define initial conditions
    y0 = computeInitialConditions()

    gstarCsvFile = "mssm_gstar.csv"
    sigVChi = 1e-20
    mChi = 200.
    GammaPhi = 1e-25
    mPhi = 5.e6
    BrPhiWIMP = 0.2
    fa = 1e11

    sol = odeint( build_Boltzmann_Equations, y0, N, args=( sigVChi, mChi, GammaPhi, BrPhiWIMP, gstarCsvFile, fa, mPhi ) )

