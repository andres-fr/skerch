#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for CLI utilities."""


import pytest
import os
from tempfile import TemporaryDirectory

from skerch.utils import REAL_DTYPES, COMPLEX_DTYPES, torch_dtype_as_str
from skerch.__main__ import PluginLoader, COMMANDS
from skerch.__main__ import matrix_shape
from skerch.__main__ import get_argparser
from skerch.__main__ import main_wrapper


# ##############################################################################
# # FIXTURES
# ##############################################################################


@pytest.fixture
def expected_plugins():
    """ """
    result = {"create_hdf5_layout_lop", "merge_hdf5", "post_bounds"}
    return result


@pytest.fixture
def all_dtypes():
    """ """
    result = [t for t in REAL_DTYPES] + [t for t in COMPLEX_DTYPES]
    return result


# ##############################################################################
# # MAIN
# ##############################################################################
def test_aux_cli(expected_plugins, monkeypatch, capsys):
    """Testing auxiliary CLI functionality.

    Note that monkeypatch (fake CLI arguments) and capsys (capture sys output)
    are default pytest fixtures.

    * Plugin loader features the expected modules
    * malformed matrix shapes
    * argparser arguments
    """
    # expected plugins
    plugins = PluginLoader.get()
    assert set(plugins) == expected_plugins, "Mismatching CLI plugins!"
    assert plugins == COMMANDS, "Mismatching CLI plugins/COMMANDS!"
    # matrix shape
    assert matrix_shape("1, 2") == (1, 2), "Unexpected matrix shape!"
    with pytest.raises(ValueError):
        _ = matrix_shape("1, 2, 3")  # too many dims
    with pytest.raises(ValueError):
        _ = matrix_shape("1")  #  too little dims
    with pytest.raises(ValueError):
        _ = matrix_shape("1, 0")  #  empty dims
    with pytest.raises(ValueError):
        _ = matrix_shape("-1, 1")  #  negative dims
    with pytest.raises(ValueError):
        _ = matrix_shape("1, 2.0")  #  float dims
    with pytest.raises(ValueError):
        _ = matrix_shape("1, abc")  #  non-numeric dims
    # argparser arguments
    parser = get_argparser()
    args = parser.parser_args()
    breakpoint()


def test_main_cli(all_dtypes, capsys):
    """Various formal tests for synthetic matrices.

    * xxx

    we want basically to simulate the integration test here, with the
    hope that it
    """
    # help string
    main_wrapper(["-h"])
    helpstr = capsys.readouterr().out
    assert "usage: pytest" in helpstr, "Unexpected -h output!"
    #
    # skerch_main(
    #     ["post_bounds", "--apost_n=30", "--apost_err=0.5", "--is_complex"]
    # )
