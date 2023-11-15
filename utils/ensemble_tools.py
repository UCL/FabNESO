""" A module to create input directories and encode 
configfurations for FabNESO
"""

import os
import shutil
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

    #Make the base directory
    if not makeDirectory(sweepPath,destructive): return

    # Set the initial value of the scanned parameter to the lower limit
    paraVal = scanRange[0]  

    for i in range(nDirs):
        newDir = "{0}/SWEEP/{2}{1}/".format(sweepPath,i,outdirPrefix)
        #Make the directory
        os.makedirs(newDir)
        #If we're copying files, do so
        if copyFiles: copyDirContents(newDir,copyDir)
        # Now we edit the parameter file for our
        # template scan if we're doing that
        if os.path.isfile(newDir+editFile) and parameterToScan:
            editParameter(newDir+editFile,parameterToScan,paraVal)
        #iterate paraVal
        paraVal+=(scanRange[1]-scanRange[0])/float(nDirs-1)


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
    """ Copy the contents of copyDir -> dirPath """
    
    for f in os.listdir(copyDir): shutil.copy(copyDir+f,dirPath)

def makeDirectory(directoryName,
                  destructive):
    """ Make the directory tree at directoryName. """
    
    if os.path.isdir(directoryName):
        if destructive: shutil.rmtree(directoryName)
        else:
            print("Path already exists and "
                  "not in destructive mode! Returning")
            return 0
    os.makedirs(directoryName)
    return 1

def encodeConditionsFile(inFileName,
                         paramDict):
    """ Encode a configuration file with a dict of input values """

    
    for paramName in paramDict.keys(): editParameter(inFileName,
                                                     paramName,
                                                     paramDict[paramName])
        
def editParameter(inFileName,
                  param,
                  val):
    """ Edit a single parameter in the configuration 
    file to the desired value """
    
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
        
