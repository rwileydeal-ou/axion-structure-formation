import numpy as np
from boltzmann import data
from boltzmann import boltzmann
from boltzmann.eqns import util


# define the moduli masses we want to use
mPhis = [
    8.e4,
    9.e4,
    1.e5,
    2.e5,
    3.e5,
    4.e5,
    5.e5,
    6.e5,
    7.e5,
    8.e5,
    9.e5,
    1.e6,
    2.e6,
    3.e6,
    4.e6,
    5.e6,
]
mPhis.reverse()

# define the branching fraction we want to use
br = 0.05

# define the effective coupling in the modulus decay width
# c for Gamma_phi = (c / 48pi) * mphi^3/mP^2
c = 25. + 2.5


# pull WIMP masses and cross sections from file
wimp_data = util.readCrossSectionMassPairsFromCSV( "fermiLAT_TauTau.csv" )

# NOTE: ALL DIMENSIONFUL VALUES SHOULD HAVE BASE UNITS OF GEV
for mPhi in mPhis:
    print( mPhi)
    Gamma_Phi = c * np.power( mPhi, 3. ) / ( 48. * np.pi * np.power( data.mPlanck, 2. ) )

    for wimp in wimp_data:
        mDM = wimp[0]
        # convert from cm^3 /s to GeV^-2
        crossSection = wimp[1] * ( 1./3. * 1.e17)

        boltzmann.solveBoltzmannEquations(
            mass_Modulus = mPhi,
            mass_WIMP    = mDM,
            crossSection_WIMP  = crossSection,
            decayWidth_Modulus = Gamma_Phi,
            branchRatio_ModulusToWIMP = br,
            fa           = 1e11,
            thetaI       = 3.113,
            temp_Reheat  = 1e12,
            gstarCsvFile = "mssm_gstar.csv",
            outputCsv    = "testOutput.csv"
        )
