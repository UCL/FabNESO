""" A helper script that will create a directory tree for a sweep and at
 some point vary a parameter in the desired file. Probably using sed.

 Update docs when its done I guess? If we decide to keep this
"""

import os
import shutil
import argparse
import sys
import itertools


def createDirTree(sweepPath,
                  nDirs,
                  destructive,
                  copyDir,
                  editFile,
                  parameterToScan,
                  scanRange,
                  outdirPrefix):
    """ Create a directory tree in the sweepPath """

    copyFiles = os.path.isdir(copyDir)
    if not copyFiles: print ("No correct copy dir found, just "
                             "creating the tree for now!")
    
    if os.path.isdir(sweepPath):
        if destructive: shutil.rmtree(sweepPath)
        else:
            print("Path already exists and not in destructive mode! Returning")
            return

        # Set the initial value of the scanned parameter to the lower limit
    paraVal = scanRange[0]  

    for i in range(nDirs):
        newDir = "{0}/SWEEP/{2}{1}/".format(sweepPath,i,outdirPrefix)
        #Make the directory
        os.makedirs(newDir)
        #If we're copying files, do so
        if copyFiles:
            for f in os.listdir(copyDir): shutil.copy(copyDir+f,newDir)
        # Now we edit the parameter file for our
        # template scan if we're doing that
        if os.path.isfile(newDir+editFile) and parameterToScan:
            editParameter(newDir+editFile,parameterToScan,paraVal)
        #iterate paraVal
        paraVal+=(scanRange[1]-scanRange[0])/float(nDirs-1)

# Make a multi-dimensional scan of several parameters
def createDictSweep(sweepPath,
                    nDirs,
                    destructive,
                    copyDir,
                    editFile,
                    parameterDict):

    """ Use a dictionary with each parameter's high and low to create
    a multi-dimensional sweep directory """
    
    print("{0} parameters found, requiring {1} divisions per parameter. "
          "Creating {2} configurations..."
          .format(len(parameterDict.keys()),
                  nDirs,
                  nDirs**len(parameterDict.keys())))

    #Calculate all of the parameter values and put them in a dict
    scanPoints = {}
    listAll = []
    for parameter in parameterDict.keys():
        paramValue = parameterDict[parameter][0]
        scanPoints[parameter] = []
        for i in range(nDirs):
            scanPoints[parameter].append(paramValue)
            listAll.append("{0}_{1}".format(parameter,i))
            paramValue += ((parameterDict[parameter][1] -
                            parameterDict[parameter][0]) / float(nDirs-1))

    # Create the combinations of these parameters. 
    combinations = list(itertools.combinations(listAll,
                                               len(parameterDict.keys())))
    allConfigs = []
    for comb in combinations:
        recordComb = True
        for i in range(len(comb) - 1):
            searchTerm = comb[i][:len(comb[i])-comb[i][::-1].find("_")]
            for j in range(i+1,len(comb)):
                if comb[j].startswith(searchTerm):
                    recordComb = False
                    break
        if recordComb: allConfigs.append(comb)

    # If destructive, let's delete the whole tree if it already exists
    if destructive and os.path.isdir(sweepPath): shutil.rmtree(sweepPath)
    # Check whether the copy directory exists
    copyFiles = os.path.isdir(copyDir)
    # Now loop over each combination, create a directory, copy the
    # configs and conditions, and edit the configuation to contain
    # the correct parameters.
    for comb in allConfigs:
        dirName = "-".join(comb)
        dirPath = "{0}/SWEEP/{1}".format(sweepPath,dirName)
        makeDirectory(dirPath,destructive)
        if copyFiles: copyDirContents(dirPath,copyDir)
        # Create parameter map for conditions encoding
        parameters = {}
        for parameter in parameterDict.keys():
            paramIndex = [item for item in comb
                          if item.startswith(parameter)][0].split("_")[-1]
            parameters[parameter] = scanPoints[parameter][int(paramIndex)]
        if editFile: encodeConditionsFile(dirPath+"/"+editFile,parameters)
    return 0

def copyDirContents(dirPath,
                    copyDir):
    for f in os.listdir(copyDir): shutil.copy(copyDir+f,dirPath)

def makeDirectory(directoryName,
                  destructive):
    if os.path.isdir(directoryName):
        if destructive: shutil.rmtree(directoryName)
        else:
            print("Path already exists and "
                  "not in destructive mode! Returning")
            return
    os.makedirs(directoryName)

# Encode a configuration file with a dict of input values
def encodeConditionsFile(inFileName,
                         paramDict):
    
    for paramName in paramDict.keys(): editParameter(inFileName,
                                                     paramName,
                                                     paramDict[paramName])
        
# Edit a single parameter in the configuration file to the desired value
def editParameter(inFileName,
                  param,
                  val):
    
    print("Edit {1} : {0}".format(val,param))
    data = []
    with open(inFileName,'r') as inFile:
        data = inFile.readlines()

    newData = []
    for line in data:
        if param in line:
            line  = line.split("=")[0] + "= {0} </P>\n".format(val)
        newData.append(line)

    with open(inFileName,"w") as outFile:
        for line in newData: outFile.writelines(line)
        
def main():

    # Make the argument parser
    parser = argparse.ArgumentParser(
        prog="makeSweepDir",
        description="Makes a sweep directory for FabNeso"
    )
    parser.add_argument("--sweepPath",
                        dest="sweepPath",
                        help="The path of the sweep directory",
                        default="test_path")
    parser.add_argument("--nDivs",
                        dest="nDivs",
                        help="Number of divisions",
                        type=int,
                        default=5)
    parser.add_argument("--destructive","-d",
                        help="Deletes the previous tree if it already exists",
                        action='store_true')
    parser.add_argument("--copyDir",
                        help="Copy contents of this dir to the sweep dirs",
                        default="plugins/FabNeso/config_files/toCopy/")
    parser.add_argument("--editFile",
                        help="Template a parameter in this file",
                        default="two_stream_conditions.xml")
    parser.add_argument("--paraToTemplate",
                        help="The parameter in the config "
                        "file to template for the scan",
                        default="")
    parser.add_argument("--scanMin",
                        help="Lower limit of the parameter scan",
                        type=float,
                        default=0)
    parser.add_argument("--scanMax",
                        help="Upper limit of the parameter scan",
                        type=float,
                        default=0)
    parser.add_argument("--parameterDict",
                        help="An input dict of the parameters to be scanned.",
                        default="")
    parser.add_argument("--dirPrefix",
                        help="Prefix to call output dir",
                        default="d")
    args = parser.parse_args()

    if args.parameterDict:
        # If this has been specified, we'll automatically create
        # the directory tree as a multidimensional scan of these points
        parameterDict = eval(args.parameterDict)
        # Probably insert here a check that the dict is of the correct
        # format before we start running the execution?
        createDictSweep(args.sweepPath,args.nDivs,args.destructive,
                        args.copyDir,args.editFile,parameterDict)
        
        sys.exit(0)
                        
    createDirTree(args.sweepPath,args.nDivs,args.destructive,
                  args.copyDir,args.editFile,args.paraToTemplate,
                  [args.scanMin,args.scanMax],args.dirPrefix)

if __name__ == "__main__":
    main()
