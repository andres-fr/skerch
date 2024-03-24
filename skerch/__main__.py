#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Command Line Interface interaction with ``skerch``.

This script acts in connection with the :mod:`.cli` submodule.
"""


import argparse
import pkgutil
import shutil
import sys
import tempfile
from unittest.mock import patch

from . import INNER_FMT, LO_FMT, RO_FMT, cli


# ##############################################################################
# # HELPERS
# ##############################################################################
def ensure_dirpath(dirpath=None):
    """Produces a valid directory path if none given.

    If ``dirpath`` is not None, returns ``dirpath``.
    If ``dirpath`` is None, prompts user to accept creating a random temporary
    directory. If user accepts directory is created and its path returned,
    otherwise process is exited.
    """
    if dirpath is None:
        dirpath = tempfile.mkdtemp()
        ans = input(f"No dirpath provided! proceed with {dirpath}? (y/N): ")
        if ans.lower() != "y":
            shutil.rmtree(dirpath)
            sys.exit(0)
    return dirpath


class PluginLoader:
    """Plugin helper that loads functionality from :cvar:`PLUGIN_MODULE`."""

    PLUGIN_MODULE = cli
    ENTRY_POINT = "main"

    @classmethod
    def get(cls):
        """Loads all plugins from :mod:`.cli`.

        :returns: A dictionary in the form ``plugin_name: plugin_fn``, where
          each name corresponds to a file in the :mod:`.cli` submodule, and
          the corresponding function is the :cvar:`ENTRY_POINT` method in that
          file.
        """
        result = {}
        modname = cls.PLUGIN_MODULE.__name__
        for sub in pkgutil.iter_modules(cls.PLUGIN_MODULE.__path__):
            subname = sub.name
            # ridiculously obscure line to allow submodules from __main__,
            # when they make relative imports themselves.
            submod = __import__(
                f"{modname}.{subname}",
                globals(),
                locals(),
                # level=1,
                fromlist=[""],
            )
            result[subname] = getattr(submod, cls.ENTRY_POINT)
        #
        return result


# ##############################################################################
# # GLOBALS
# ##############################################################################
COMMANDS = PluginLoader.get()  # parse the .cli module and retrieve functions


# ##############################################################################
# # PARSE CLI
# ##############################################################################
def matrix_shape(shape):
    """Checks and parses given string into a shape.

    If given ``shape`` is a string with two positive, coma-separated integers
    like '100,200', returns a tuple in the form ``(100, 200)``. Otherwise
    raises an ``AssertionError``.
    """
    h, w = shape.split(",")
    h, w = int(h), int(w)
    assert h > 0, "Height must be positive!"
    assert w > 0, "Width must be positive!"
    return (h, w)


def get_argparser():
    """Argument parser for ``skerch`` CLI interaction.

    :returns: An ``ArgumentParser`` that first accepts a positional argument
      with the ``command`` to be executed (must be one of the dict keys
      returned by :meth:`PluginLoader.get`), and then accepts optional
      arguments that may be applicable to that particular command. Call
      with ``-h`` for details.

    Usage examples::
      python -m skerch post_bounds --apost_n=30 --apost_err=0.75
      python -m skerch prio_hpars --shape=100,200 --budget=12345
      python -m skerch create_hdf5_layout --dtype float64 \
        --shape 123,234 --outer 30 --inner 60
    """
    parser = argparse.ArgumentParser(description="skerch CLI")
    parser.add_argument(
        "command",
        metavar="[COMMAND]",
        action="store",
        type=str,
        choices=set(COMMANDS),
        help=f"Determines which functionality to run: {set(COMMANDS)}",
    )
    # a priori parameter estimation
    parser.add_argument(
        "--shape",
        type=matrix_shape,
        default=(100, 200),
        help="Matrix shape in the form 'height,width' as positive integers.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=10_000,
        help="Total number of scalar measurements that we can store.",
    )
    parser.add_argument(
        "--sym",
        action="store_true",  # default is false
        help="If given, matrix is assumed to be Hermitian.",
    )
    # a posteriori error bounds
    parser.add_argument(
        "--apost_n",
        default=30,
        type=int,
        help="Number of a-posteriori measurements intended.",
    )
    parser.add_argument(
        "--apost_err",
        default=0.5,
        type=float,
        help="A-posteriori target error, from 0 (no error) to 1 (0x - 2x).",
    )
    # create HDF5 layout
    parser.add_argument(
        "--hdf5dir",
        default=None,
        type=str,
        help="Directory to create the HDF5 layout.",
    )
    parser.add_argument(
        "--outer", default=100, type=int, help="Number of outer measurements."
    )
    parser.add_argument(
        "--inner", default=200, type=int, help="Number of inner measurements."
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        type=str,
        help="Datatype of HDF5 layout to be created. Must be empty",
    )
    # merge virtual HDF5 into monolithic
    parser.add_argument(
        "--in_path",
        default=None,
        type=str,
        help="Input path for the file to be processed.",
    )
    parser.add_argument(
        "--out_path",
        default=None,
        type=str,
        help="Input path for the file to be processed.",
    )
    parser.add_argument(
        "--ok_flag",
        default=None,
        type=str,
        help="If given, all HDF5 flags are checked to equal this.",
    )
    parser.add_argument(
        "--delete_subfiles",
        action="store_true",
        help="If given, HDF5 subfiles are deleted upon merging to monolithic.",
    )
    return parser


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
def main_wrapper(cli_args=None):
    """Wrapped :func:`main` function.

    This wrapper gives the option to programmatically mock CLI arguments if
    desired. When running normally, e.g. via ``python -m ...``, it does
    nothing.

    :param cli_args: If none given, CLI arguments from ``sys.argv`` are left
      untouched. Otherwise, ``sys.argv[1:]`` will be mocked to have the given
      values.

    The following snippet has the same effect as running
    ``python -m skerch post_bounds --apost_n=30 --apost_err=0.75`` from the
    CLI::

      from skerch.__main__ import main_wrapper as skm
      skm(["post_bounds", "--apost_n=30", "--apost_err=0.75"])
    """
    if cli_args is None:
        main()
    else:
        cli_args = [sys.argv[0]] + list(cli_args)
        with patch.object(sys, "argv", cli_args):
            try:
                main()
            except SystemExit:
                # ArgParser exits with '-h', we don't want that
                # https://stackoverflow.com/a/58367457
                pass


def main():
    """Entry point for ``python -m skerch``."""
    # parse args and check that action is recognized
    parser = get_argparser()
    args = parser.parse_args()
    cmd = args.command
    assert cmd in COMMANDS, f"Unknown command! {cmd}"
    main = COMMANDS[cmd]
    # a-priori estimation of hyperparameters
    if cmd == "prio_hpars":
        shape = args.shape
        budget = args.budget
        sym = args.sym
        main(shape, budget, sym)
    # a-posteriori error probability bounds
    elif cmd == "post_bounds":
        n = args.apost_n
        err = args.apost_err
        main(n, err)
    # create HDF5 layout
    elif cmd == "create_hdf5_layout":
        dirpath = ensure_dirpath(args.hdf5dir)
        shape = args.shape
        with_ro = not args.sym  # if matrix Hermitian, no need for right outer
        outer = args.outer
        inner = args.inner
        dtype = args.dtype
        main(
            dirpath,
            shape,
            dtype,
            outer,
            inner,
            LO_FMT,
            RO_FMT,
            INNER_FMT,
            with_ro,
        )
    # merge
    elif cmd == "merge_hdf5":
        inpath = args.in_path
        outpath = args.out_path
        flag = args.ok_flag
        delete_sub = args.delete_subfiles
        main(inpath, outpath, flag, delete_sub)


if __name__ == "__main__":
    main()
