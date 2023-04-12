from glob import glob
import re
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

def make_mWIMP_vs_oh2_vs_BR_plot(baseDir, outFile):
    brs = [
        1e-2,
        1e-3,
        1e-4,
        1e-5,
        1e-6,
        1e-7,
        1e-8,
        1e-9,
        1e-10,
        1e-11
    ]

    for i in range(0,10):
        data = readOutputFile(f'{baseDir}/branching_{i}.csv')
        mPhi = []
        mWIMP = []
        sigvWIMP = []
        oH2_WIMP = []
        oH2_Axion = []

        for d in data:
            mPhi.append( d[0] )
            mWIMP.append( d[1] )
            sigvWIMP.append( d[2] )
            oH2_WIMP.append( d[3] )

        br = brs[i]
        plt.plot(  
            mWIMP,
            oH2_WIMP,
            label=f'Br={br}'
        )

    plt.plot(  
        [10., 10000.],
        [0.12, 0.12],
        label=f'$\Omega h^2 = 0.12$',
        linestyle='dashed',
        c='black'
    )
    plt.xlim( 10., 10000. )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$m_{\text{WIMP}}$ (GeV)')
    plt.ylabel(r'$\Omega h^2$')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.))
    plt.tight_layout()
    plt.savefig(outFile)
    plt.clf()
    plt.close("all")

def make_mPhi_vs_oh2_vs_BR_plot(posWIMP, baseDir, outFile):
    brs = [
        1e-2,
        1e-3,
        1e-4,
        1e-5,
        1e-6,
        1e-7,
        1e-8,
        1e-9,
        1e-10,
        1e-11
    ]

    outDirs = sorted(glob(f'{baseDir}/*/', recursive = True), key=lambda x:float(re.findall("-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?",x)[0]))
    for i in range(0,10):
        mPhi = []
        oH2_WIMP = []
        for outDir in outDirs:
            data = readOutputFile(f'{outDir}/branching_{i}.csv')
            mPhi.append( data[posWIMP][0] )
            oH2_WIMP.append( data[posWIMP][3] )

        br = brs[i]
        plt.plot(  
            mPhi,
            oH2_WIMP,
            label=f'Br={br}'
        )

    plt.plot(  
        [1e5, 5e6],
        [0.12, 0.12],
        label=f'$\Omega h^2 = 0.12$',
        linestyle='dashed',
        c='black'
    )
    plt.xlim( 1e5, 5e6 )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$m_{\phi}$ (GeV)')
    plt.ylabel(r'$\Omega h^2$')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.))
    plt.tight_layout()
    plt.savefig(outFile)
    plt.clf()
    plt.close("all")

make_mWIMP_vs_oh2_vs_BR_plot("outputs/5e6", "branching_UpperBound_100percent.png")
make_mWIMP_vs_oh2_vs_BR_plot("outputs/1e5", "branching_LowerBound_100percent.png")
make_mPhi_vs_oh2_vs_BR_plot( 0, "outputs/", "branching_mPhi_mWIMP_10GeV_100percent.png" )
make_mPhi_vs_oh2_vs_BR_plot( 9, "outputs/", "branching_mPhi_mWIMP_100GeV_100percent.png" )
make_mPhi_vs_oh2_vs_BR_plot( 18, "outputs/", "branching_mPhi_mWIMP_1000GeV_100percent.png" )
make_mPhi_vs_oh2_vs_BR_plot( 27, "outputs/", "branching_mPhi_mWIMP_10000GeV_100percent.png" )



exit()
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
