import re
from urllib import request
import numpy as np
import scipy.special
from boltzmann import data
from boltzmann.eqns import util

################################################################
#               Primative helper functions                     #
################################################################


# this method calculates rhoEquilibrium for WIMP
# see e.g. Eq.A.8 in arXiv: 1110.2491
def compute_rhoEquilibrium( temp, mass_WIMP ):
    g = 2

    if temp == 0.:
        return 0.
    
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
    w_Modulus = util.compute_equationOfState_CohOsc( mass=mass_Modulus, hubble=hubble )

    jacMod = [ 0., 0., 0., 0. ]

    # hubble dilution
    dEdN = -3. * rho_Modulus * ( 1. + w_Modulus )

    # decay
    dEdN -= decayWidth_Modulus * rho_Modulus / hubble

    jacMod[ 0 ] = - ( 3. * ( 1. + w_Modulus ) + decayWidth_Modulus / hubble )

    return [ dEdN, jacMod ]

def build_WIMP_energyDensity_eqn( inputData, energyDensities, temp ):
    crossSection_WIMP = inputData.crossSection_WIMP
    mass_WIMP = inputData.mass_WIMP
    decayWidth_Modulus = inputData.decayWidth_Modulus
    branchRatio_ModulusToWIMP = inputData.branchRatio_ModulusToWIMP
    hubble = energyDensities.hubble
    rho_WIMP = energyDensities.rho_WIMP
    rhoEQ_WIMP = energyDensities.rhoEQ_WIMP
    rho_Modulus = energyDensities.rho_Modulus

    w_WIMP = util.compute_equationOfStateWIMP( temp=temp, mass_WIMP=mass_WIMP )

    jacWIMP = [ 0., 0., 0., 0. ]

    # hubble dilution
    dEdN = -3. * rho_WIMP * ( 1. + w_WIMP )
    jacWIMP[1] = - 3. * ( 1. + w_WIMP ) 

    # annihilations
    # annihilation term is numerically stiff - but equilibrium is an attractor
    # make approximation that WIMP is in equilibrium until close to freeze-out, Tf ~ mDM / 20
    if temp <= 20. * 1.5 * ( mass_WIMP / 20. ):
        dEdN -= ( np.power(rho_WIMP, 2.) - np.power(rhoEQ_WIMP, 2.) ) * crossSection_WIMP / ( hubble * mass_WIMP ) 
        jacWIMP[1] -= 2. * rho_WIMP * crossSection_WIMP / ( hubble * mass_WIMP )

    # injections
    dEdN += decayWidth_Modulus * rho_Modulus * branchRatio_ModulusToWIMP / hubble 
    jacWIMP[0] = decayWidth_Modulus * branchRatio_ModulusToWIMP / hubble

    return [ dEdN, jacWIMP ]

def build_axion_numberDensity_eqn( mass_Axion, energyDensities ):
    jacAxion = [ 0., 0., 0., 0. ]
    w_Axion = util.compute_equationOfState_CohOsc( mass=mass_Axion, hubble=energyDensities.hubble )

    # hubble dilution 
    dndN = -3. * energyDensities.n_Axion * ( 1. + w_Axion )

    jacAxion[2] = -3. * ( 1. + w_Axion )
    return [ dndN, jacAxion ]

def build_radiation_energyDensity_eqn( inputData, energyDensities, temp ):
    crossSection_WIMP = inputData.crossSection_WIMP
    mass_WIMP = inputData.mass_WIMP
    decayWidth_Modulus = inputData.decayWidth_Modulus
    branchRatio_ModulusToWIMP = inputData.branchRatio_ModulusToWIMP
    hubble = energyDensities.hubble
    rho_Radiation = energyDensities.rho_Radiation
    rho_WIMP = energyDensities.rho_WIMP
    rhoEQ_WIMP = energyDensities.rhoEQ_WIMP
    rho_Modulus = energyDensities.rho_Modulus
    jacRad = [ 0., 0., 0., 0. ]

    # hubble dilution
    dEdN = -4. * rho_Radiation
    jacRad[3] = -4.

    # annihilations - again assume WIMP is in equilibrium until close to freeze-out
    if temp <= 20. * 1.5 * ( mass_WIMP / 20. ):
        dEdN += ( np.power(rho_WIMP, 2.) - np.power(rhoEQ_WIMP, 2.) ) * crossSection_WIMP / ( mass_WIMP * hubble )
        jacRad[1] += 2. * rho_WIMP * crossSection_WIMP / ( hubble * mass_WIMP )

    # decays
    dEdN += decayWidth_Modulus * rho_Modulus * ( 1. - branchRatio_ModulusToWIMP ) / hubble
    jacRad[0] = decayWidth_Modulus * ( 1. - branchRatio_ModulusToWIMP ) / hubble

    return [ dEdN, jacRad ]


# this helper method computes the zeroth order Boltzmann equations
def build_zerothOrder_equations( 
    inputData,
    energyDensities,
    mass_Axion, 
    temp
):
    dRhoPhi = build_modulus_energyDensity_eqn( inputData=inputData, energyDensities=energyDensities )
    dRhoChi = build_WIMP_energyDensity_eqn( inputData=inputData, energyDensities=energyDensities, temp=temp )
    dnAx = build_axion_numberDensity_eqn( mass_Axion, energyDensities=energyDensities )
    dRhoRad = build_radiation_energyDensity_eqn( inputData=inputData, energyDensities=energyDensities, temp=temp )

    eqs = [ dRhoPhi[0], dRhoChi[0], dnAx[0], dRhoRad[0] ]
    jac = np.row_stack(( dRhoPhi[1], dRhoChi[1], dnAx[1], dRhoRad[1] ))

    return [ eqs, jac ]

