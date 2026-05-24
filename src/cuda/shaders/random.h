#ifndef GPU_RANDOM_H
#define GPU_RANDOM_H

// XORShift128+ — fast, good quality, matches CPU SmallRng seed behavior
struct RngState {
    unsigned long long s0, s1;
};

// Initialize from a 32-bit pixel seed (deterministic)
__device__ inline RngState rng_init(unsigned int seed) {
    // SplitMix64 seeding
    unsigned long long z = seed;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31);
    RngState s;
    s.s0 = z;
    s.s1 = z ^ 0x9e3779b97f4a7c15ULL;
    return s;
}

// Returns uniform float in [0, 1)
__device__ inline float rng_uniform(RngState* state) {
    unsigned long long s1 = state->s0;
    const unsigned long long s0 = state->s1;
    state->s0 = s0;
    s1 ^= s1 << 23;
    state->s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    // Return lower 32 bits as float in [0,1)
    return (float)(state->s1 >> 11) * 0x1.0p-53f;
}

#endif // GPU_RANDOM_H
