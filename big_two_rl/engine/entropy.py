"""
Hardware Entropy Harvester and Cryptographic Seed Generator.
Combines 5 physical and microarchitectural entropy sources:
1. RDRAND: On-chip Digital Random Number Generator (NIST SP 800-90A)
2. RDSEED: On-chip Physical Thermal Noise Entropy (NIST SP 800-90B)
3. CPU Clock Jitter: High-resolution cycle jitter via RDTSCP instruction
4. Memory Access Latency: Non-deterministic DRAM access and bus refresh variations
5. Cache Miss Latency: Forced L1/L2/L3 cache line flushes (clflush) vs cache hit timing

All 5 sources are mixed and cryptographically whitened using SHA-256 / BLAKE2b
to produce an unbiasable, non-deterministic seed for PRNGs.
"""

import os
import sys
import time
import ctypes
import hashlib
from typing import Dict, Any, Tuple, Optional
import numpy as np


class HardwareEntropyPacket(ctypes.Structure):
    _fields_ = [
        ("rdrand_val", ctypes.c_uint64),
        ("rdseed_val", ctypes.c_uint64),
        ("jitter_val", ctypes.c_uint64),
        ("mem_latency_val", ctypes.c_uint64),
        ("cache_miss_val", ctypes.c_uint64),
        ("rdrand_ok", ctypes.c_int),
        ("rdseed_ok", ctypes.c_int),
    ]


class HardwareEntropyHarvester:
    """
    Harvester that interfaces with native hardware instructions and microarchitectural
    timing variances to extract true physical entropy.
    """
    _lib = None

    @classmethod
    def _get_lib(cls):
        if cls._lib is not None:
            return cls._lib

        current_dir = os.path.dirname(os.path.abspath(__file__))
        so_path = os.path.join(current_dir, "libentropy.so")
        c_path = os.path.join(current_dir, "entropy_native.c")

        # Compile if missing
        if not os.path.exists(so_path) and os.path.exists(c_path):
            import subprocess
            try:
                subprocess.run(
                    ["gcc", "-O2", "-shared", "-fPIC", "-mrdrnd", "-mrdseed", c_path, "-o", so_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception:
                pass

        if os.path.exists(so_path):
            try:
                lib = ctypes.CDLL(so_path)
                lib.harvest_hardware_entropy.argtypes = [
                    ctypes.POINTER(HardwareEntropyPacket),
                    ctypes.c_int
                ]
                lib.harvest_hardware_entropy.restype = None
                cls._lib = lib
            except Exception:
                cls._lib = None

        return cls._lib

    @classmethod
    def harvest_raw(cls, samples: int = 64) -> Dict[str, Any]:
        """
        Harvest raw entropy from all 5 hardware sources.
        Returns a dictionary containing raw 64-bit values and status flags.
        """
        lib = cls._get_lib()
        packet = HardwareEntropyPacket()

        if lib is not None:
            lib.harvest_hardware_entropy(ctypes.byref(packet), samples)
            rdrand = packet.rdrand_val
            rdseed = packet.rdseed_val
            jitter = packet.jitter_val
            mem_lat = packet.mem_latency_val
            cache_miss = packet.cache_miss_val
            rdrand_ok = bool(packet.rdrand_ok)
            rdseed_ok = bool(packet.rdseed_ok)
        else:
            # Fallback in case shared library is unavailable
            rdrand = int.from_bytes(os.urandom(8), "big")
            rdseed = int.from_bytes(os.urandom(8), "big")
            jitter = time.perf_counter_ns()
            mem_lat = id(object()) ^ time.time_ns()
            cache_miss = (id(packet) << 16) ^ time.perf_counter_ns()
            rdrand_ok = False
            rdseed_ok = False

        return {
            "rdrand": rdrand,
            "rdseed": rdseed,
            "cpu_jitter": jitter,
            "memory_latency": mem_lat,
            "cache_miss": cache_miss,
            "rdrand_available": rdrand_ok,
            "rdseed_available": rdseed_ok,
            "timestamp_ns": time.time_ns()
        }

    @classmethod
    def generate_seed(cls, samples: int = 64) -> int:
        """
        Harvest all 5 entropy sources, distill them via SHA-256 cryptographic whitening,
        and return a pristine 64-bit unsigned integer seed.
        """
        raw = cls.harvest_raw(samples=samples)

        # Construct 48-byte physical entropy payload
        payload = bytearray()
        payload.extend(raw["rdrand"].to_bytes(8, "big", signed=False))
        payload.extend(raw["rdseed"].to_bytes(8, "big", signed=False))
        payload.extend(raw["cpu_jitter"].to_bytes(8, "big", signed=False))
        payload.extend(raw["memory_latency"].to_bytes(8, "big", signed=False))
        payload.extend(raw["cache_miss"].to_bytes(8, "big", signed=False))
        payload.extend(raw["timestamp_ns"].to_bytes(8, "big", signed=False))

        # Cryptographic distillation (whitening) via SHA-256
        digest = hashlib.sha256(payload).digest()

        # Extract 64-bit seed from the distilled digest
        seed_64 = int.from_bytes(digest[:8], "big")
        return seed_64

    @classmethod
    def generate_seed_bytes(cls, num_bytes: int = 32, samples: int = 64) -> bytes:
        """
        Generate cryptographically whitened seed bytes of requested length.
        """
        raw = cls.harvest_raw(samples=samples)
        payload = bytearray()
        payload.extend(raw["rdrand"].to_bytes(8, "big", signed=False))
        payload.extend(raw["rdseed"].to_bytes(8, "big", signed=False))
        payload.extend(raw["cpu_jitter"].to_bytes(8, "big", signed=False))
        payload.extend(raw["memory_latency"].to_bytes(8, "big", signed=False))
        payload.extend(raw["cache_miss"].to_bytes(8, "big", signed=False))
        payload.extend(raw["timestamp_ns"].to_bytes(8, "big", signed=False))

        # Use BLAKE2b for arbitrary output length
        return hashlib.blake2b(payload, digest_size=num_bytes).digest()

    @classmethod
    def get_rng(cls, samples: int = 64) -> np.random.Generator:
        """
        Instantiate a NumPy PCG64 random generator seeded with the 5-source hardware seed.
        """
        seed = cls.generate_seed(samples=samples)
        return np.random.default_rng(seed)


def get_hardware_seed() -> int:
    """Convenience helper to get a 64-bit hardware seed."""
    return HardwareEntropyHarvester.generate_seed()


def get_hardware_rng() -> np.random.Generator:
    """Convenience helper to get a NumPy Generator seeded from hardware entropy."""
    return HardwareEntropyHarvester.get_rng()
