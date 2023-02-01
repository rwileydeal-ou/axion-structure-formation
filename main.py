import numpy as np
from boltzmann import data
from boltzmann import boltzmann

# NOTE: ALL DIMENSIONFUL VALUES SHOULD HAVE BASE UNITS OF GEV
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
for mPhi in mPhis:
    br = 0.05

    c = 25. + 2.5
    Gamma_Phi = c * np.power( mPhi, 3. ) / ( 48. * np.pi * np.power( data.mPlanck, 2. ) )
    print( mPhi )
    boltzmann.solveBoltzmannEquations(
        mass_Modulus=mPhi,
        mass_WIMP=200., 
        crossSection_WIMP=1.93e-08,
        decayWidth_Modulus = Gamma_Phi,
        branchRatio_ModulusToWIMP = br,
        fa = 1e11,
        thetaI = 3.113,
        temp_Reheat=1e12,
        gstarCsvFile = "mssm_gstar.csv"
    )
