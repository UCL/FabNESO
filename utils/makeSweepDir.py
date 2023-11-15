""" Run ther create directory methods of the enseble_tools module
"""

from ensemble_tools import createDirTree,createDictSweep

import argparse
import sys

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
