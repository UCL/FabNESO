# A helper script that will create a directory tree for a sweep and at some point vary a parameter in the desired file. Probably using sed.
#
# Update docs when its done I guess? If we decide to keep this

import os,shutil,argparse

#Delete any existing versions of this tree and makes the new one. Maybe this is undesirable?
def createDirTree(sweepPath,nDirs,destructive,copyDir,editFile,parameterToScan,scanRange,outdirPrefix):
    copyFiles = os.path.isdir(copyDir)
    if not copyFiles: print ("No correct copy dir found, just creating the tree for now!")
    if os.path.isdir(sweepPath):
        if destructive: shutil.rmtree(sweepPath)
        else:
            print("Path already exists and not in destructive mode! Returning")
            return
    #Set the initial value of the scanned parameter to the lower limit
    paraVal = scanRange[0]  
    for i in range(nDirs):
        newDir = "{0}/SWEEP/{2}{1}/".format(sweepPath,i,outdirPrefix)
        #Make the directory
        os.makedirs(newDir)
        #If we're copying files, do so
        if copyFiles:
            for f in os.listdir(copyDir): shutil.copy(copyDir+f,newDir)
        #Now we edit the parameter file for our template scan if we're doing that
        if os.path.isfile(newDir+editFile) and parameterToScan: editParameter(newDir+editFile,parameterToScan,paraVal)
        #iterate paraVal
        paraVal+=(scanRange[1]-scanRange[0])/float(nDirs-1)
        
def editParameter(inFileName,param,val):
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

    parser = argparse.ArgumentParser(
        prog="makeSweepDir",
        description="Makes a sweep directory for FabNeso"
    )
    parser.add_argument("--sweepPath",dest="sweepPath",help="The path of the sweep directory",default="test_path")
    parser.add_argument("--nDivs",dest="nDivs",help="Number of divisions",type=int,default=5)
    parser.add_argument("--destructive","-d",help="Deletes the previous tree if it already exists",action='store_true')
    parser.add_argument("--copyDir",help="Copy contents of this dir to the sweep dirs",default="plugins/FabNeso/config_files/toCopy/")
    parser.add_argument("--editFile",help="Template a parameter in this file",default="two_stream_conditions.xml")
    parser.add_argument("--paraToTemplate",help="The parameter in the config file to template for the scan",default="")
    parser.add_argument("--scanMin",help="Lower limit of the parameter scan",type=float,default=0)
    parser.add_argument("--scanMax",help="Upper limit of the parameter scan",type=float,default=0)
    parser.add_argument("--dirPrefix",help="Prefix to call output dir",default="d")
    args = parser.parse_args()
    
    createDirTree(args.sweepPath,args.nDivs,args.destructive,args.copyDir,args.editFile,args.paraToTemplate,[args.scanMin,args.scanMax],args.dirPrefix)

if __name__ == "__main__":
    main()
