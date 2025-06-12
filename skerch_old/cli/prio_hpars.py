#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""A-priori sketch hyperparameters.

CLI plugin to provide optimal number of measurements for a given budget and
linear operator.
"""

from ..a_priori import a_priori_hyperparams


# ##############################################################################
# # ENTRY POINT
# ##############################################################################
def main(shape, budget, hermitian=False):
    """See :func:`.a_priori.a_priori_hyperparams`."""
    outer, inner = a_priori_hyperparams(
        shape, budget, complex_data=False, hermitian=hermitian
    )
    print("Shape:", shape)
    print("Budget:", budget)
    print("Hermitian:", hermitian)
    print("  Outer measurements:", outer)
    print("  Inner measurements:", inner)
    print(
        f"[CHECK]: {outer} * {sum(shape)} + {inner}**2 =",
        (outer * sum(shape)) + inner**2,
    )
    print(
        f"[CHECK]: {outer + 1} * {sum(shape)} + {inner}**2 =",
        ((outer + 1) * sum(shape)) + inner**2,
    )
