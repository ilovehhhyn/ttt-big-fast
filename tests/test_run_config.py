"""The command-line rules of ttt.run that decide a run's parameter sets: per-token rates must
land in a slow set, an arm without one refuses them, and only arm F takes a prime width."""

from __future__ import annotations

import pytest

from ttt.run import ARMS, resolve_prime_intermediate, resolve_slow_spec


def test_token_rates_join_the_slow_set_of_an_arm_that_has_one():
    assert resolve_slow_spec(ARMS["C"]["slow"], token_rates=True) == ARMS["C"]["slow"] + ("token_rate",)
    assert resolve_slow_spec(ARMS["D"]["slow"], token_rates=True) == ARMS["D"]["slow"]


def test_token_rates_off_leaves_every_arm_unchanged():
    for arm in sorted(ARMS):
        assert resolve_slow_spec(ARMS[arm]["slow"], token_rates=False) == ARMS[arm]["slow"], arm


@pytest.mark.parametrize("arm", ["A", "B"])
def test_token_rates_are_refused_without_a_slow_set(arm: str) -> None:
    with pytest.raises(AssertionError, match="token-rates.*slow"):
        resolve_slow_spec(ARMS[arm]["slow"], token_rates=True)


def test_arm_f_requires_a_prime_width_and_other_arms_refuse_one():
    assert resolve_prime_intermediate("F", 2048) == 2048
    with pytest.raises(AssertionError, match="--prime-intermediate is required"):
        resolve_prime_intermediate("F", None)
    for arm in ("A", "B", "C", "D", "E"):
        assert resolve_prime_intermediate(arm, None) is None
        with pytest.raises(AssertionError, match="no effect"):
            resolve_prime_intermediate(arm, 2048)
