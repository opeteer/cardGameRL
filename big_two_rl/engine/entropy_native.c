/*
 * Native Hardware Entropy Harvester for Big Two RL.
 * Combines 5 hardware-level sources:
 * 1. RDRAND: On-chip Digital Random Number Generator (NIST SP 800-90A)
 * 2. RDSEED: On-chip True Physical Thermal Noise Entropy (NIST SP 800-90B)
 * 3. CPU Clock Jitter: Cycle-to-cycle instruction pipeline & frequency scaling timing variance
 * 4. Memory Access Latency: DRAM bus refresh & uncached memory access variance
 * 5. Cache Miss Latency: Explicit L1/L2/L3 cache line evictions (clflush) vs cache hit delta
 */

#include <stdint.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <stdlib.h>
#include <string.h>

#define FNV_OFFSET_BASIS_64 0xcbf29ce484222325ULL
#define FNV_PRIME_64        0x100000001b3ULL

// 1. RDRAND Harvester
int get_rdrand64(uint64_t *val) {
    if (!val) return 0;
    for (int retry = 0; retry < 50; retry++) {
        unsigned char ok;
        __asm__ volatile("rdrand %0; setc %1" : "=r"(*val), "=qm"(ok));
        if (ok) return 1;
        __builtin_ia32_pause();
    }
    return 0;
}

// 2. RDSEED Harvester (Raw Physical Entropy)
int get_rdseed64(uint64_t *val) {
    if (!val) return 0;
    for (int retry = 0; retry < 200; retry++) {
        unsigned char ok;
        __asm__ volatile("rdseed %0; setc %1" : "=r"(*val), "=qm"(ok));
        if (ok) return 1;
        __builtin_ia32_pause();
    }
    return 0;
}

// 3. CPU Clock Jitter Harvester
// Measures microscopic cycle variance across instruction sequences using RDTSCP
uint64_t get_cpu_jitter(int samples) {
    if (samples <= 0) samples = 64;
    uint64_t hash = FNV_OFFSET_BASIS_64;
    uint32_t aux;

    for (int i = 0; i < samples; i++) {
        uint64_t t1 = __rdtscp(&aux);
        
        // Microarchitectural operations sensitive to branch prediction and CPU core scheduling
        volatile int x = (i * 37) + 11;
        for (int k = 0; k < 8; k++) {
            x = (x << 3) ^ (x >> 5) ^ (k * 29);
        }

        uint64_t t2 = __rdtscp(&aux);
        uint64_t delta = t2 - t1;

        // Fold delta into rolling hash
        hash = (hash ^ delta) * FNV_PRIME_64;
    }

    return hash;
}

// 4. Memory Access Latency Harvester
// Measures non-deterministic DRAM bus latency and refresh cycles across a large memory pool
uint64_t get_memory_latency(int samples) {
    if (samples <= 0) samples = 64;
    size_t sz = 16 * 1024 * 1024; // 16 MB buffer (exceeds L1/L2 cache)
    volatile uint8_t *buf = (volatile uint8_t *)malloc(sz);
    if (!buf) return 0;

    // Seed page tables
    for (size_t i = 0; i < sz; i += 4096) {
        buf[i] = (uint8_t)(i & 0xFF);
    }

    uint64_t hash = 0x84222325ULL;
    size_t idx = 0;
    size_t large_prime = 524309; // Large prime stride across memory pages
    uint32_t aux;

    for (int i = 0; i < samples; i++) {
        idx = (idx + large_prime + (hash & 0xFFFF)) % sz;

        uint64_t t1 = __rdtscp(&aux);
        volatile uint8_t val = buf[idx];
        uint64_t t2 = __rdtscp(&aux);

        uint64_t delta = t2 - t1;
        uint64_t entry = delta | ((uint64_t)val << 32);
        hash = (hash ^ entry) * FNV_PRIME_64;
    }

    free((void *)buf);
    return hash;
}

// 5. Cache Miss Latency Harvester
// Explicitly forces cache line misses via clflush and measures the hit vs miss cycle difference
uint64_t get_cache_miss_latency(int samples) {
    if (samples <= 0) samples = 64;
    size_t line_size = 64;
    size_t num_lines = 512;
    size_t total_size = num_lines * line_size;
    
    volatile uint8_t *buf = (volatile uint8_t *)aligned_alloc(64, total_size);
    if (!buf) return 0;

    memset((void *)buf, 0x5A, total_size);
    uint64_t hash = 0x9e3779b97f4a7c15ULL;
    uint32_t aux;

    for (int i = 0; i < samples; i++) {
        size_t line_idx = (i * 13 + (hash & 0x1F)) % num_lines;
        void *line_ptr = (void *)(buf + line_idx * line_size);

        // 1. Flush cache line to force a miss on subsequent read
        _mm_clflush(line_ptr);
        __builtin_ia32_mfence();

        // 2. Measure cache miss reload latency (from DRAM)
        uint64_t t1 = __rdtscp(&aux);
        volatile uint8_t miss_val = *(volatile uint8_t *)line_ptr;
        uint64_t t2 = __rdtscp(&aux);
        uint64_t miss_cycles = t2 - t1;

        // 3. Measure immediate cache hit reload latency (from L1 cache)
        uint64_t t3 = __rdtscp(&aux);
        volatile uint8_t hit_val = *(volatile uint8_t *)line_ptr;
        uint64_t t4 = __rdtscp(&aux);
        uint64_t hit_cycles = t4 - t3;

        // Combine miss, hit, and read values
        uint64_t diff = (miss_cycles << 16) ^ hit_cycles ^ ((uint64_t)miss_val << 48);
        hash = (hash ^ diff) * FNV_PRIME_64;
    }

    free((void *)buf);
    return hash;
}

// Collect raw entropy packet of 40 bytes (5 x 64-bit values)
typedef struct {
    uint64_t rdrand_val;
    uint64_t rdseed_val;
    uint64_t jitter_val;
    uint64_t mem_latency_val;
    uint64_t cache_miss_val;
    int rdrand_ok;
    int rdseed_ok;
} HardwareEntropyPacket;

void harvest_hardware_entropy(HardwareEntropyPacket *pkt, int samples) {
    if (!pkt) return;
    memset(pkt, 0, sizeof(HardwareEntropyPacket));
    pkt->rdrand_ok = get_rdrand64(&pkt->rdrand_val);
    pkt->rdseed_ok = get_rdseed64(&pkt->rdseed_val);
    pkt->jitter_val = get_cpu_jitter(samples);
    pkt->mem_latency_val = get_memory_latency(samples);
    pkt->cache_miss_val = get_cache_miss_latency(samples);
}
