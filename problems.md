CPU Reference Prover:

    Need to figure out what the public parameters ( A, B1, L, H, B2) and inputs (w, ca, cb, cc) are and how to print them.
        - Looked at the docs: https://coinlist.co/build/coda/pages/problem-07-groth16-prover-challenges which provided insight into the parameters and inputs

    Need to understand the compute_H() function where all the FFTs happen
        - Help with the libfqfft tutorials described here: https://github.com/TalDerei/libfqfft/tree/master/tutorials

    FFT section is very complicated right now, and does not account for the major bottleneck
        - For now focus on MSM (as it accounts for 70-80% of the prover runtime in SNARK systems)