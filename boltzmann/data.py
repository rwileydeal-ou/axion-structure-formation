import dataclasses


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
    thetaI: float
    temp_Reheat: float
    gstarCsvFile: str

@dataclasses.dataclass
class EnergyDensities:
    rho_Modulus: float
    rho_WIMP: float
    rho_Radiation: float
    n_Axion: float
    hubble: float
    rhoEQ_WIMP: float = 0.

