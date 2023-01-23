import numpy as np
from boltzmann.eqns import gstr
from boltzmann.eqns import zero
from boltzmann.eqns import util
from boltzmann import data

################################################################
#        Wrapped helper functions - Initial Conditions         #
################################################################

def compute_hubble_initialCondition( 
    inputData,
    gstar
):
    hubble0 = np.sqrt(
        np.power( np.pi, 2. ) * gstar / 90.
    ) * np.power( inputData.temp_Reheat, 2. ) / data.mPlanck
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
    amplitude = data.mPlanck
    rho0_mod = 0.5 * np.power( inputData.mass_Modulus * amplitude, 2. )
    return rho0_mod

def compute_axion_initialCondition(
    inputData,
    mass_Axion
):
    # domain wall number, in DFSZ case nDW=6
    nDW = 6
    fTheta = np.power( np.log( np.exp(1.) / (1. - np.power(inputData.thetaI / np.pi, 2.) ) ), 7./6. )
    amplitude_SQR = 1.44 * np.power( inputData.fa * inputData.thetaI / nDW, 2. ) * fTheta

    n0_Ax = 0.5 * mass_Axion * amplitude_SQR
    return n0_Ax

def compute_zerothOrder_initialConditions(
    inputData
):
    # read gstar once
    gstar = gstr.readGstarFromCSV( gstarCsvFile=inputData.gstarCsvFile, temp=inputData.temp_Reheat )

    hubble0 = compute_hubble_initialCondition( inputData=inputData, gstar=gstar )
    rho0_Radiation = compute_radiation_initialCondition( inputData=inputData, gstar=gstar )

    # for cases of interest, WIMP starts in equilibrium - just calculate rho_EQ at TR
    rho0_WIMP = zero.compute_rhoEquilibrium( temp=inputData.temp_Reheat, mass_WIMP=inputData.mass_WIMP )

    rho0_Modulus = compute_modulus_initialCondition( inputData=inputData, gstar=gstar )
    massAxion = util.axionMass( fa=inputData.fa, temp=inputData.temp_Reheat )
    n0_Axion = compute_axion_initialCondition( inputData=inputData, mass_Axion=massAxion )

    return [rho0_Modulus, rho0_WIMP, n0_Axion, rho0_Radiation]

