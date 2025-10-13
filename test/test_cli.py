#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for CLI utilities."""


import pytest
import os
from tempfile import TemporaryDirectory

from skerch.a_posteriori import apost_error_bounds
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

    * Plugin loader features the expected modules
    * malformed matrix shapes
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


def test_main_cli(all_dtypes, capsys):
    """Various formal tests for synthetic matrices.

    Note that monkeypatch (fake CLI arguments) and capsys (capture sys output)
    are default pytest fixtures.

    * xxx


    we want basically to simulate the integration test here, with the
    hope that it
    """
    # help string
    main_wrapper(["-h"])
    help_str = capsys.readouterr().out
    assert "usage: pytest" in help_str, "Unexpected -h output!"
    # post bounds
    apost, err = 30, 0.5
    main_wrapper(
        [
            "post_bounds",
            f"--apost_n={apost}",
            f"--apost_err={err}",
            "--is_complex",
        ]
    )
    bounds_str = capsys.readouterr().out
    bounds = apost_error_bounds(apost, err, is_complex=True)
    assert str(bounds) == bounds_str[:-1], "Malformed post_bounds output?"
    # create_hdf5_layout_lop (leave it open)
    tmpdir = TemporaryDirectory()
    main_wrapper(
        [
            "create_hdf5_layout_lop",
            f"--hdf5dir={tmpdir.name}",
            "--lop_shape=100,200",
            "--dtype=complex128",
            "--partsize=10",
            "--lo=30",
            "--ro=30",
            "--inner=60",
        ]
    )
    h5_names = os.listdir(tmpdir.name)
    assert (
        len([n for n in h5_names if "ALL" in n]) == 3
    ), "Wrong ALL CLI files?"
    assert (
        len([n for n in h5_names if "leftouter" in n]) == 4
    ), "Wrong leftouter CLI files?"
    assert (
        len([n for n in h5_names if "rightouter" in n]) == 4
    ), "Wrong rightouter CLI files?"
    assert (
        len([n for n in h5_names if "inner" in n]) == 7
    ), "Wrong inner CLI files?"
    # try to create on existing dirpath:error
    with pytest.raises(RuntimeError):
        main_wrapper(
            [
                "create_hdf5_layout_lop",
                f"--hdf5dir={tmpdir.name}",
                "--lop_shape=100,200",
                "--dtype=complex128",
                "--partsize=10",
                "--lo=30",
                "--ro=30",
                "--inner=60",
            ]
        )
    # merge_hdf5 and delete subfiles
    main_wrapper(
        [
            "merge_hdf5",
            "--delete_subfiles",
            "--ok_flag=initialized",
            "--in_path",
            os.path.join(tmpdir.name, "leftouter_ALL.h5"),
        ]
    )
    main_wrapper(
        [
            "merge_hdf5",
            "--delete_subfiles",
            "--ok_flag=initialized",
            "--in_path",
            os.path.join(tmpdir.name, "rightouter_ALL.h5"),
        ]
    )
    main_wrapper(
        [
            "merge_hdf5",
            "--delete_subfiles",
            "--ok_flag=initialized",
            "--in_path",
            os.path.join(tmpdir.name, "inner_ALL.h5"),
        ]
    )
    h5_names_merged = os.listdir(tmpdir.name)
    assert h5_names_merged == [
        "rightouter_ALL.h5",
        "inner_ALL.h5",
        "leftouter_ALL.h5",
    ], "Wrong CLI merged flies?"
    tmpdir.cleanup()
