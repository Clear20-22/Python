"""
Baby-step Giant-step Algorithm for Discrete Logarithms.

Reference: https://en.wikipedia.org/wiki/Baby-step_giant-step

Given a base, a target integer, and a modulus, this algorithm finds the smallest
non-negative integer exponent such that:
    base^exponent = target (mod modulus)

The algorithm operates in O(sqrt(modulus)) time and O(sqrt(modulus)) space using
a meet-in-the-middle approach. This implementation also supports non-coprime
moduli by repeatedly factoring out gcd(base, modulus).
"""

from __future__ import annotations

import math


def baby_step_giant_step(base: int, target: int, modulus: int) -> int | None:
    """
    Find the smallest non-negative integer exponent such that:
        base^exponent = target (mod modulus)

    Parameters:
        base: The base integer (positive).
        target: The target residue (non-negative).
        modulus: The modulus integer (greater than 1).

    Returns:
        The smallest non-negative integer exponent solving the congruence,
        or None if no solution exists.

    Raises:
        ValueError: If modulus <= 1, base <= 0, or target < 0.

    Time Complexity: O(sqrt(modulus))
    Space Complexity: O(sqrt(modulus))

    Examples:
    >>> baby_step_giant_step(2, 3, 5)
    3
    >>> baby_step_giant_step(3, 13, 17)
    4
    >>> baby_step_giant_step(5, 1, 7)
    0
    >>> baby_step_giant_step(2, 8, 16)
    3
    >>> baby_step_giant_step(2, 7, 16) is None
    True
    >>> baby_step_giant_step(6, 2, 7) is None
    True
    >>> baby_step_giant_step(10, 20, 30) is None
    True
    >>> baby_step_giant_step(5, 123456789, 1000000007)
    981640996
    >>> baby_step_giant_step(0, 5, 7)
    Traceback (most recent call last):
        ...
    ValueError: Base must be a positive integer (got 0).
    >>> baby_step_giant_step(3, -1, 7)
    Traceback (most recent call last):
        ...
    ValueError: Target must be a non-negative integer (got -1).
    >>> baby_step_giant_step(2, 5, 1)
    Traceback (most recent call last):
        ...
    ValueError: Modulus must be greater than 1 (got 1).
    """
    if modulus <= 1:
        msg = f"Modulus must be greater than 1 (got {modulus})."
        raise ValueError(msg)
    if base <= 0:
        msg = f"Base must be a positive integer (got {base})."
        raise ValueError(msg)
    if target < 0:
        msg = f"Target must be a non-negative integer (got {target})."
        raise ValueError(msg)

    base %= modulus
    target %= modulus

    if target == 1:
        return 0

    # Handle non-coprime case by reducing the congruence
    factor_accumulator = 1
    eliminated_steps = 0
    common_divisor = math.gcd(base, modulus)

    while common_divisor > 1:
        if target % common_divisor != 0:
            return None
        target //= common_divisor
        modulus //= common_divisor
        factor_accumulator = (factor_accumulator * (base // common_divisor)) % modulus
        eliminated_steps += 1
        if target == factor_accumulator % modulus:
            return eliminated_steps
        common_divisor = math.gcd(base, modulus)

    # Standard Baby-step Giant-step for coprime base and modulus
    step_size = math.isqrt(modulus) + 1

    # Baby-step phase:
    # Compute and store target * base^baby_index (mod modulus) for:
    #     0 <= baby_index < step_size.
    # To find the smallest solution exponent = giant_index * step_size - baby_index,
    # we record the largest baby_index if duplicate values appear.
    baby_steps: dict[int, int] = {}
    current_baby_value = target
    for baby_index in range(step_size):
        baby_steps[current_baby_value] = baby_index
        current_baby_value = (current_baby_value * base) % modulus

    # Giant-step phase:
    # Check if factor_accumulator * (base^step_size)^giant_index matches any baby step.
    giant_step_multiplier = pow(base, step_size, modulus)
    current_giant_value = (factor_accumulator * giant_step_multiplier) % modulus

    for giant_index in range(1, step_size + 1):
        if current_giant_value in baby_steps:
            baby_index = baby_steps[current_giant_value]
            return giant_index * step_size - baby_index + eliminated_steps
        current_giant_value = (current_giant_value * giant_step_multiplier) % modulus

    return None


if __name__ == "__main__":
    import doctest

    doctest.testmod()
