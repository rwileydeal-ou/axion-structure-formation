from boltzmann import boltzmann

# NOTE: ALL DIMENSIONFUL VALUES SHOULD HAVE BASE UNITS OF GEV
boltzmann.solveBoltzmannEquations(
    mass_Modulus=1.e6, 
    mass_WIMP=200., 
    crossSection_WIMP=1.93e-08,
    decayWidth_Modulus = 5e-20,
    branchRatio_ModulusToWIMP = 0.1,
    fa = 1e11,
    thetaI = 3.113,
    temp_Reheat=1e12,
    gstarCsvFile = "mssm_gstar.csv"
)
