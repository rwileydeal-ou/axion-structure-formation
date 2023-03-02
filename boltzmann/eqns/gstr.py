import csv

# this method retrieves gstar from CSV file and stores in array
def pullGstarFromCSV( gstarCsvFile ):
    gstrData = []
    with open(gstarCsvFile, newline='\n', mode='r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if row[0][0] == '#':
                continue
            gstrData.append(row)
    return gstrData

# this method retrieves gstar from data (input array) for specified temperature
# if temperature does not match exact entry, interpolates between two nearest neighbors
def readGstarFromData( gstarCsvData, temp ):
    tempUpper = 0.
    gstarUpper = 0.
    for dataRow in gstarCsvData:
        if dataRow[0][0] == '#':
            continue
        # if bull's eye, just return
        if float(dataRow[0]) == temp:
            return float(dataRow[1])
        # otherwise, if temp between tempUpper and current row, interpolate
        if float(dataRow[0]) < temp:
            dT = tempUpper - float(dataRow[0])
            dGStr = gstarUpper - float(dataRow[1])                
            m = dGStr / dT
            gstr = m * ( temp - float(dataRow[0]) ) + float(dataRow[1])
            return gstr
        else:
            tempUpper = float(dataRow[0])
            gstarUpper = float(dataRow[1])

    return 225.
