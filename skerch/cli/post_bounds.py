#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""CLI plugin to provide probabilistic bounds for aposteriori error estimation.

Borrows functionaliy from :func:`skerch.a_posteriori.apost_error_bounds`.
"""


from ..a_posteriori import apost_error_bounds


# ##############################################################################
# # ENTRY POINT
# ##############################################################################
def main(num_meas, rel_err, cplx):
    """Entry point for this CLI script. See module docstring."""
    result = apost_error_bounds(num_meas, rel_err, cplx)
    print(result)
