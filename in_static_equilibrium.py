"""
Checks if a system of forces is in static equilibrium.
"""
from __future__ import annotations

from numpy import array, cos, cross, float64, radians, sin
from numpy.typing import NDArray


def polar_force(
    magnitude: float, angle: float, radian_mode: bool = False
) -> list[float]:
    """
    Resolves force along rectangular components.
    (force, angle) => (force_x, force_y)
    >>> import math
    >>> force = polar_force(10, 45)
    >>> math.isclose(force[0], 7.071067811865477)
    True
    >>> math.isclose(force[1], 7.0710678118654755)
    True
    >>> force = polar_force(10, 3.14, radian_mode=True)
    >>> math.isclose(force[0], -9.999987317275396)
    True
    >>> math.isclose(force[1], 0.01592652916486828)
    True
    """
    if radian_mode:
        return [magnitude * cos(angle), magnitude * sin(angle)]
    return [magnitude * cos(radians(angle)), magnitude * sin(radians(angle))]


def in_static_equilibrium(
    forces: NDArray[float64], location: NDArray[float64], eps: float = 10**-1
) -> bool:
    """
    Check if a system is in equilibrium.
    It takes two numpy.array objects.
    forces ==>  [
                        [force1_x, force1_y],
                        [force2_x, force2_y],
                        ....]
    location ==>  [
                        [x1, y1],
                        [x2, y2],
                        ....]
    >>> force = array([[1, 1], [-1, 2]])
    >>> location = array([[1, 0], [10, 0]])
    >>> in_static_equilibrium(force, location)
    False
    """
    # summation of moments is zero
    moments: NDArray[float64] = cross(location, forces)
    sum_moments: float = sum(moments)
    return abs(sum_moments) < eps
