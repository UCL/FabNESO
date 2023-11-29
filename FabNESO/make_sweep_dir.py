"""Makes a sweep directory for FabNESO."""

import argparse
from ast import literal_eval
from pathlib import Path

from .ensemble_tools import create_dict_sweep, create_dir_tree


def main():
    # Make the argument parser
    parser = argparse.ArgumentParser(
        description="Makes a sweep directory for FabNESO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep_path",
        help="The path to write the ",
        type=Path,
        default=Path(__file__).parent.parent / "config_files" / "two_stream_ensemble",
    )
    parser.add_argument(
        "--n_divs",
        help="Number of divisions in grid for each parameter",
        type=int,
        default=5,
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
        type=Path,
        default=Path(__file__).parent.parent / "config_files" / "two_stream",
    )
    parser.add_argument(
        "--edit_file",
        help="Template a parameter in this file",
        default="conditions.xml",
    )
    parser.add_argument(
        "--para_to_template",
        help="The parameter in the config file to template for the scan",
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
        parameter_dict = literal_eval(args.parameter_dict)
        # Check we have made a dict
        if not isinstance(parameter_dict, dict):
            msg = "Did not receive a dict as input for parameter_dict"
            raise ValueError(msg)
        # Use the dict to create a sweep directory
        create_dict_sweep(
            sweep_path=args.sweep_path,
            n_divs=args.n_divs,
            destructive=args.destructive,
            copy_dir=args.copy_dir,
            edit_file=args.edit_file,
            parameter_dict=parameter_dict,
        )

    else:
        create_dir_tree(
            sweep_path=args.sweep_path,
            n_dirs=args.n_divs,
            destructive=args.destructive,
            copy_dir=args.copy_dir,
            edit_file=args.edit_file,
            parameter_to_scan=args.para_to_template,
            scan_range=(args.scan_min, args.scan_max),
            outdir_prefix=args.dir_prefix,
        )


if __name__ == "__main__":
    main()
