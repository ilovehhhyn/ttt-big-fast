"""The command-line rules of ttt.run that decide a run's slow set: per-token rates must
land in a slow set, and an arm without one refuses them."""

from __future__ import annotations

import pytest

from ttt.run import ARMS, resolve_slow_spec


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
