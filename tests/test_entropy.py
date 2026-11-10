"""
Unit tests for the 5-Source Hardware Entropy Harvester:
- RDRAND
- RDSEED
- CPU Clock Jitter
- Memory Access Latency
- Cache Miss Latency
- SHA-256 Cryptographic Distillation
"""
import pytest
import numpy as np

from big_two_rl.engine.entropy import (
    HardwareEntropyHarvester,
    get_hardware_seed,
    get_hardware_rng
)
from big_two_rl.engine.game_state import GameState


class TestHardwareEntropy:
    def test_all_five_sources_present_and_nonzero(self):
        """Verify that all 5 hardware sources produce valid 64-bit non-zero integers."""
        raw = HardwareEntropyHarvester.harvest_raw(samples=32)

        assert "rdrand" in raw
        assert "rdseed" in raw
        assert "cpu_jitter" in raw
        assert "memory_latency" in raw
        assert "cache_miss" in raw

        # All values should be 64-bit unsigned integers
        for key in ["rdrand", "rdseed", "cpu_jitter", "memory_latency", "cache_miss"]:
            val = raw[key]
            assert isinstance(val, int), f"{key} must be an integer"
            assert 0 <= val < (1 << 64), f"{key} must be within 64-bit range"
            assert val != 0, f"{key} produced an unexpected all-zero output"

        # Check hardware support status
        assert raw["rdrand_available"] is True, "CPU must support RDRAND"
        assert raw["rdseed_available"] is True, "CPU must support RDSEED"

    def test_seed_uniqueness_no_collisions(self):
        """Verify that consecutive calls produce unique, non-repeating seeds."""
        num_seeds = 50
        seeds = [get_hardware_seed() for _ in range(num_seeds)]

        # All generated seeds must be distinct
        unique_seeds = set(seeds)
        assert len(unique_seeds) == num_seeds, "Collision detected in hardware seeds!"

    def test_hardware_rng_generation(self):
        """Verify that get_hardware_rng() creates a functional NumPy Generator."""
        rng = get_hardware_rng()
        assert isinstance(rng, np.random.Generator)

        # Draw random integers and permutation
        samples = rng.integers(0, 100, size=10)
        assert len(samples) == 10
        perm = rng.permutation(52)
        assert len(perm) == 52
        assert len(set(perm)) == 52

    def test_game_state_uses_hardware_seed_when_none(self):
        """Verify that GameState dealing works cleanly with hardware entropy."""
        gs = GameState()
        start_p = gs.reset(seed=None)
        assert start_p in [0, 1, 2, 3]

        # Check all hands have 13 unique cards
        all_cards = []
        for p in range(4):
            assert len(gs.hands[p]) == 13
            all_cards.extend(gs.hands[p])
        assert len(set(all_cards)) == 52
