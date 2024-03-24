#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""A-posteriori error bounds.

CLI plugin to provide a-posteriori error bounds for given number of measurements
and tolerance.
"""


from ..a_posteriori import a_posteriori_error_bounds


# ##############################################################################
# # ENTRY POINT
# ##############################################################################
def main(num_meas, rel_err):
    """See :func:`.a_posteriori.a_posteriori_error_bounds`."""
    result = a_posteriori_error_bounds(num_meas, rel_err)
    print(result)
