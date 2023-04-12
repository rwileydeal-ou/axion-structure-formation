import csv
import numpy as np
import scipy.special
from boltzmann.eqns import gstr as gStrReader


# this method computes axion mass based on current temperature (in GeV) and axion decay constant (in GeV)
def axionMass(fa, temp):
    m = np.power(0.078, 2.) / fa
    if temp == 0.:
        temp = 0.001
    return min( m, ( np.power(0.2 / temp, 4.) * 0.018 * m ) )

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


# this method computes the radiation temperature from the radiation energy density
# requires a CSV file containing gstar data, interpolates gstar with Temp estimations to find correct value
def temperature( rho_Radiation, gstarCsvFile ):
    gstarGuess = 100.

    temp0 = np.power( 30. * rho_Radiation / ( gstarGuess * np.power(np.pi, 2.) ), 0.25 )
    temp1 = 0.
    deltaTemp = 0.

    while ( np.abs( temp0 - temp1 ) / temp0 > 0.01 ):
        gstr = gStrReader.readGstarFromData( gstarCsvFile, temp0 )
        temp1 = np.power( 30. * rho_Radiation / ( gstr * np.power(np.pi, 2.) ), 0.25 )

        if ( np.abs( temp0 - temp1) == deltaTemp ):
            temp0 = ( temp0 + temp1 ) / 2.
        else:
            deltaTemp = np.abs( temp0 - temp1 )
            temp0 = temp1

    return temp1

def compute_equationOfStateWIMP( temp, mass_WIMP ):
    if ( temp <= mass_WIMP / 10. ):
        return 0
    elif ( temp <= mass_WIMP / 5. and temp > mass_WIMP / 10. ):
        f1 = mass_WIMP * scipy.special.kn(1, mass_WIMP / temp) / scipy.special.kn(2, mass_WIMP / temp) / temp 
        f2 = 3.
        w2 = 1 / ( f1 + f2 )
        wEST = 0. + ( w2 ) / 0.1 * ( temp / mass_WIMP - 0.1 )
        return wEST
    elif ( temp < 1. * mass_WIMP and temp > mass_WIMP / 5. ):
        # intermediate regime
        f1 = mass_WIMP * scipy.special.kn(1, mass_WIMP / temp) / scipy.special.kn(2, mass_WIMP / temp) / temp 
        f2 = 3.
        return 1 / ( f1 + f2 )
    elif ( temp >= 1. * mass_WIMP and temp < 1.5 * mass_WIMP ):
        f1 = mass_WIMP * scipy.special.kn(1, mass_WIMP / temp) / scipy.special.kn(2, mass_WIMP / temp) / temp 
        f2 = 3.
        w2 = 1 / ( f1 + f2 )
        
        wEST = w2 + ( 1./3. - w2 ) / 0.5 * ( temp / mass_WIMP - 1. )
        return wEST
    elif ( temp >= 1.5 * mass_WIMP  ):
        return 1./3.

def compute_equationOfState_CohOsc( mass, hubble ):
    if mass < hubble:
        return -1.
    return 0.


# this method retrieves list of ( mass, <sig.v> ) pairs from file
def readCrossSectionMassPairsFromCSV( csVsMassCsvFile ):
    mass_crossSection_Pairs = []
    with open(csVsMassCsvFile, newline='\n', mode='r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if row[0][0] == '#':
                continue
            mass_crossSection_Pairs.append( [ float(row[0]), float(row[1]) ] )
    return mass_crossSection_Pairs

# this method retrieves list of ( mass, <sig.v> ) pairs from file
def readInputDataFromCSV( inputCsvFile ):
    inputData = []
    with open(inputCsvFile, newline='\n', mode='r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if row[0][0] == '#':
                continue
            inputData.append( [ float(row[0]), float(row[1]), float(row[2]) ] )
    return inputData

