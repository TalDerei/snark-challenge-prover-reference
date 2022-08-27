**Reference Materials**:
- Coda + Dekrypt SNARK Challenge: Coda Workshop: https://www.youtube.com/watch?v=81uR9W5PZ5M&t=772s&ab_channel=CoinList
- Documentation and starter code: https://coinlist.co/build/coda/pages/prover#starter-code
- Cuda-Fixnum (Extended-precision modular arithmetic library): https://github.com/TalDerei/cuda-fixnum

Exisiting Implementations:
1. Don't take full advantage of parallelism, i.e. use of GPUs
2. Don't use all known optimization techniques for elliptic curve arithmetic

SNARK Provers have two main algorithms:
- Elliptic-Curve Multiexponentiation (MSM)
    - large map-reduce
    - inputs 
        - each input is a pair consisting of a "scalar" (768-bit integer), and a "curve point" (pair of 768-bit integers)
    - map: apply function to each input
        - "curve points" (i.e. big ints) can be combined with a function we'll call pointAdd
        - let's write P + Q for pointAdd(P, Q)
        - our map function will be scale(s: Scalar, P: Point): Point, with scale(s, P) = P + ... + P, s many times
        - checkout out "double-and-add" algorithm trick
    - reduce: combine map results
        - pointAdd(P: Point, Q: Point): Point 
    - computation
        - whole computation looks like scale(s1, P1) + ... + scale(sn, Pn), where + is pointAdd
    - output
- Fast-Fourier Transform (FFTs)

It's important to note that there is G1 and G2 multi-exponentiation

Q. How can we make "map" (scale function) efficient?

1. There are different ways to compute pointAdd with different efficiency characteristics.
2. Can do precomputation to speed things up at the cost of using more memory ("windowing").
3. If we combine multiple "map" steps together, we can unlock additional optimizations by sharing computation between them ("batching").

**Tutorials:**

*1. Field Arithmetic*

Tutorial starter code implements Field Arithmetic: multiply together arrays of elements of a prime-order field using montgomery arithmetic.

Montgomery representation is alternative way of representing elements of F_q for more efficient multiplication. Let q be MNT4753.q/MNT6753.q and R = 2^768. The Montgomery representation of the nubmer x (e.g. 5) is (xR) mod q. This number then is represented as a little-endian length 12 array of 64-bit integers, where each element in the array is called a limb. 

Usually we're used to working 32-bit/64-bit integers. With SNARK provers, the integers are much larger. The integers are 753 bits and represented using arrays of native integers. For example, we could represent them using an array of 12 64-bit integers (since 12 * 64 = 768 > 753). And instead of computing mod 2^753, we'll compute mod q where q is either MNT4753.q or MNT6753.q. Each element of such an array is called a "limb". For example, we would say that we can represent a 2^768 bit integer using 12 64-bit limbs.



​
 
