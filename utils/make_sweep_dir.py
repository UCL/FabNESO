""" Run ther create directory methods of the enseble_tools module
"""

from ensemble_tools import create_dir_tree, create_dict_sweep

import argparse
import sys


def main():
    # Make the argument parser
    parser = argparse.ArgumentParser(
        prog="makeSweepDir", description="Makes a sweep directory for FabNeso"
    )
    parser.add_argument(
        "--sweep_path",
        dest="sweep_path",
        help="The path of the sweep directory",
        default="test_path",
    )
    parser.add_argument(
        "--n_divs", dest="n_divs", help="Number of divisions", type=int, default=5
    )
    parser.add_argument(
        "--destructive",
        "-d",
        help="Deletes the previous tree if it already exists",
        action="store_true",
    )
    parser.add_argument(
        "--copy_dir",
        help="Copy contents of this dir to the sweep dirs",
        default="plugins/FabNeso/config_files/toCopy/",
    )
    parser.add_argument(
        "--edit_file",
        help="Template a parameter in this file",
        default="two_stream_conditions.xml",
    )
    parser.add_argument(
        "--para_to_template",
        help="The parameter in the config " "file to template for the scan",
        default="",
    )
    parser.add_argument(
        "--scan_min", help="Lower limit of the parameter scan", type=float, default=0
    )
    parser.add_argument(
        "--scan_max", help="Upper limit of the parameter scan", type=float, default=0
    )
    parser.add_argument(
        "--parameter_dict",
        help="An input dict of the parameters to be scanned.",
        default="",
    )
    parser.add_argument("--dir_prefix", help="Prefix to call output dir", default="d")
    args = parser.parse_args()

    if args.parameter_dict:
        # If this has been specified, we'll automatically create
        # the directory tree as a multidimensional scan of these points
        parameter_dict = eval(args.parameter_dict)
        # Probably insert here a check that the dict is of the correct
        # format before we start running the execution?
        create_dict_sweep(
            args.sweep_path,
            args.n_divs,
            args.destructive,
            args.copy_dir,
            args.edit_file,
            parameter_dict,
        )

        sys.exit(0)

    create_dir_tree(
        args.sweep_path,
        args.n_divs,
        args.destructive,
        args.copy_dir,
        args.edit_file,
        args.para_to_template,
        [args.scan_min, args.scan_max],
        args.dir_prefix,
    )


if __name__ == "__main__":
    main()
