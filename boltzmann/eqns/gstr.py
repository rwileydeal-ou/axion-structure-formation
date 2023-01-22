import csv

# this method retrieves gstar from file for specified temperature
# if temperature does not match exact entry, interpolates between two nearest neighbors
def readGstarFromCSV( gstarCsvFile, temp ):
    with open(gstarCsvFile, newline='\n', mode='r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        tempUpper = 0.
        gstarUpper = 0.
        for row in spamreader:
            if row[0][0] == '#':
                continue
            # if bull's eye, just return
            if float(row[0]) == temp:
                return float(row[1])
            # otherwise, if temp between tempUpper and current row, interpolate
            if float(row[0]) < temp:
                dT = tempUpper - float(row[0])
                dGStr = gstarUpper - float(row[1])                
                m = dGStr / dT
                gstr = m * ( temp - float(row[0]) ) + float(row[1])
                return gstr
            else:
                tempUpper = float(row[0])
                gstarUpper = float(row[1])

    return 225.
