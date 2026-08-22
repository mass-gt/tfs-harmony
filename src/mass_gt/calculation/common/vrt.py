from numba import njit
from scipy.stats import genextreme
from typing import Any, List

import numpy as np


@njit
def draw_choice_mcs(
    cum_probs: np.ndarray,
    seed: int = None
) -> int:
    '''
    Trek een keuze uit een array met cumulatieve kansen o.b.v. Monte Carlo simulatie.
    '''
    if seed is not None:
        np.random.seed(seed)
        np.random.seed(np.random.randint(10000000))

    num_alt: int = len(cum_probs)

    rand: float = np.random.rand()
    for alt in range(num_alt):
        if cum_probs[alt] >= rand:
            return alt

    raise Exception(
        '\nError in function "draw_choice_mcs", random draw ' +
        'was outside range of cumulative probability distribution.'
    )


def draw_choice_max_util(
    probs: np.ndarray,
    seed: int = None,
    all_alts_avail: bool = True
) -> int:
    """
    Trek een keuze uit een array met kansen o.b.v. het maximum nut met expliciete error term.
    """
    if seed is not None:
        np.random.seed(seed)
        np.random.seed(np.random.randint(1000000))

    num_alt = len(probs)
    error_term: np.ndarray = genextreme.rvs(0.0, size=num_alt)

    if all_alts_avail:
        choice: int = np.argmax(np.log(probs) + error_term)
    else:
        where_avail: List[int] = np.where(probs > 0.0)[0]
        choice = where_avail[np.argmax(np.log(probs[where_avail]) + error_term[where_avail])]

    return choice


def draw_choice_distr(
    distr: np.ndarray,
    seed: int = None
) -> float:
    """
    Trek een keuze uit een array met waardes met uniforme kansen.
    """
    if seed is not None:
        np.random.seed(seed)
        np.random.seed(np.random.randint(1000000))

    return distr[np.random.randint(len(distr))]