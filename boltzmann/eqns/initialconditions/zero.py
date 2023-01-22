import numpy as np
from boltzmann.eqns import gstr
from boltzmann.eqns import zero
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
    gstar
):
    return 0.

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
    n0_Axion = compute_axion_initialCondition( inputData=inputData, gstar=gstar )

    return [rho0_Modulus, rho0_WIMP, n0_Axion, rho0_Radiation]

