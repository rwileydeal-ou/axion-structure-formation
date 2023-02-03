import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command


# this method retrieves data from file
def readOutputFile( outputFile ):
    data = []
    with open(outputFile, newline='\n', mode='r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if row[0][0] == '#':
                continue
            data.append( [ float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]) ] )
    return data

data = readOutputFile("testOutput.csv")
mPhi = []
mWIMP = []
sigvWIMP = []
oH2_WIMP = []
oH2_Axion = []

for d in data:
    if d[0] < 5000000.:
        break
    mPhi.append( d[0] )
    mWIMP.append( d[1] )
    sigvWIMP.append( d[2] )
    oH2_WIMP.append( d[3] )

plt.plot(  
    mWIMP,
    oH2_WIMP
)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$m_{\text{WIMP}}$ (GeV)')
plt.ylabel(r'$\Omega h^2$')
plt.savefig("test.png")
