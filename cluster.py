import sys
import numpy as np
from boltzmann import data
from boltzmann import boltzmann
from boltzmann.eqns import util


def doWork(mPhi, fa, percentDM, br, outFile):
    # define the effective coupling in the modulus decay width
    # c for Gamma_phi = (c / 48pi) * mphi^3/mP^2
    c = 25. + 2.5

    # pull WIMP masses and cross sections from file
    wimp_data = util.readCrossSectionMassPairsFromCSV( "fermiLAT_TauTau.csv" )

    # NOTE: ALL DIMENSIONFUL VALUES SHOULD HAVE BASE UNITS OF GEV
    Gamma_Phi = c * np.power( mPhi, 3. ) / ( 48. * np.pi * np.power( data.mPlanck, 2. ) )

    for wimp in wimp_data:
        mDM = wimp[0]

        # bail early 
        if abs(mDM - 200.) > 0.1:
            continue

        # convert from cm^3 /s to GeV^-2
        crossSection = wimp[1] * ( 1./3. * 1.e17) / np.power( percentDM, 2. ) # divide by xi^2 since constraint is fitted from data

        boltzmann.solveBoltzmannEquations(
            mass_Modulus = mPhi,
            mass_WIMP    = mDM,
            crossSection_WIMP  = crossSection,
            decayWidth_Modulus = Gamma_Phi,
            branchRatio_ModulusToWIMP = br,
            fa           = fa,
            thetaI       = 3.113,
            temp_Reheat  = 1e12,
            gstarCsvFile = "mssm_gstar.csv",
            outputCsv    = outFile
        )

inScript = sys.argv[1]
outFile = sys.argv[2]

inData = util.readInputDataFromCSV( inScript )

for d in inData:
    doWork( d[0], d[1], 0.1, 0.005, outFile )

