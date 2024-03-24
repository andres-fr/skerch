#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest configuration.

This configuration module is implicitly parsed by ``pytest``.
"""


import pytest


# ##############################################################################
# # CLI ARGUMENTS
# ##############################################################################
def str_of_unique_ints(x):
    """Typechecker function for pytest CLI arguments.

    Input ``x`` is expected to be a string with integers in the form '0 1 2 3'.
    If that is the case, returns a list ``[0, 1, 2, 3]``. Otherwise, a
    ``pytest.UsageError`` is raised.
    """
    # handle the empty case
    if isinstance(x, str) and len(x) == 0:
        return []
    # otherwise, test for list of unique integers
    sep = " "
    try:
        result = [int(elt) for elt in x.split(sep)]
        assert len(result) == len(set(result))  # repeated values?
    except (ValueError, AssertionError) as err:
        raise pytest.UsageError(
            "cmdopt must be a string with unique spase-separated integers!"
        ) from err
    #
    return result


def pytest_addoption(parser):
    """Adds CLI option to the ``pytest`` command.

    Usage example::

      pytest ... --seeds='1 2 3 4'
    """
    parser.addoption(
        "--seeds",
        action="store",
        default="0 1 -12345 12345 479915",  # 0b1110101001010101011
        help="Space-separated string of integers to use as seeds.",
        type=str_of_unique_ints,
    )
    parser.addoption(
        "--quick",
        action="store_true",
        help="If given, less tests will be run",
    )
    parser.addoption(
        "--skip_toomanyfiles",
        action="store_true",
        help="If given, test_too_many_files will do nothing (but still run).",
    )
