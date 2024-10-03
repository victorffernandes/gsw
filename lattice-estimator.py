###### original paper security ######
q_ = 2 ** 10
params = LWE.Parameters(n=10, q=q_, Xs=ND.Uniform(1, q_),Xe=ND.Uniform(0, 10), m= 1000)
params


LWE.estimate(params)

# dual                 :: rop: ≈2^38.2, mem: 41, m: 64, β: 40, d: 74, ↻: 1, tag: dual


###### at least 32-bit security ######
q_ = 2 ** 12
params = LWE.Parameters(n=35, q=q_, Xs=ND.Uniform(1, q_),Xe=ND.Uniform(0, 10), m= 1000)
params


LWE.estimate(params)

# usvp                 :: rop: ≈2^36.4, red: ≈2^36.4, δ: 1.012006, β: 51, d: 52, tag: usvp
# bdd                  :: rop: ≈2^38.6, red: ≈2^38.6, svp: ≈2^22.1, β: 40, η: 2, d: 62, tag: bdd
# bdd_mitm_hybrid      :: rop: ≈2^39.9, red: ≈2^39.9, svp: ≈2^14.5, β: 40, η: 2, ζ: 0, |S|: 1, d: 152, prob: 1, ↻: 1, tag: hybrid
# dual                 :: rop: ≈2^41.0, mem: ≈2^17.9, m: 116, β: 40, d: 151, ↻: 1, tag: dual

###### at least 64-bit security ######
q_ = 2 ** 22
params = LWE.Parameters(n=260, q=q_, Xs=ND.Uniform(1, q_), Xe=ND.Uniform(0, 100), m= 1000)
params


LWE.estimate(params)

# usvp                 :: rop: ≈2^65.5, red: ≈2^65.5, δ: 1.008324, β: 123, d: 655, tag: usvp
# bdd                  :: rop: ≈2^63.7, red: ≈2^63.0, svp: ≈2^62.2, β: 114, η: 144, d: 669, tag: bdd
# dual                 :: rop: ≈2^67.1, mem: ≈2^37.5, m: 435, β: 125, d: 695, ↻: 1, tag: dual
# dual_hybrid          :: rop: ≈2^73.2, red: ≈2^73.2, guess: ≈2^53.5, β: 149, p: 3, ζ: 0, t: 0, β': 167, N: ≈2^33.4, m: 260


###### at least 128-bit security ######
q_ = 2 ** 26
params = LWE.Parameters(n=600, q=q_, Xs=ND.Uniform(1, q_), Xe=ND.Uniform(0, 100), m= 1000)
params


LWE.estimate(params)

# bkw                  :: rop: ≈2^158.7, m: ≈2^147.2, mem: ≈2^148.2, b: 12, t1: 3, t2: 15, ℓ: 11, #cod: 305, #top: 0, #test: 37, tag: coded-bkw
# usvp                 :: rop: ≈2^131.6, red: ≈2^131.6, δ: 1.004258, β: 363, d: 840, tag: usvp
# bdd                  :: rop: ≈2^128.0, red: ≈2^126.9, svp: ≈2^127.1, β: 346, η: 379, d: 828, tag: bdd
# dual                 :: rop: ≈2^137.3, mem: ≈2^88.1, m: 496, β: 380, d: 871, ↻: 1, tag: dual
# dual_hybrid          :: rop: ≈2^140.2, red: ≈2^140.1, guess: ≈2^136.7, β: 374, p: 3, ζ: 10, t: 0, β': 377, N: ≈2^83.0, m: 375
