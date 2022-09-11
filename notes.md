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

*Stage 0: Field Arithmetic*

Problem: Use field arithmetic (called modular multiplication) to multiply together arrays of elements of a prime-order field using montgomery arithmetic.

Montgomery representation is alternative way of representing elements of F_q for more efficient multiplication. Let q be MNT4753.q/MNT6753.q and R = 2^768. The Montgomery representation of the nubmer x (e.g. 5) is (xR) mod q. This number then is represented as a little-endian length 12 array of 64-bit integers, where each element in the array is called a limb. 

Usually we're used to working 32-bit/64-bit integers. With SNARK provers, the integers are much larger. The integers are 753 bits and represented using arrays of native integers. For example, we could represent them using an array of 12 64-bit integers (since 12 * 64 = 768 > 753). And instead of computing mod 2^753, we'll compute mod q where q is either MNT4753.q or MNT6753.q. Each element of such an array is called a "limb". For example, we would say that we can represent a 2^768 bit integer using 12 64-bit limbs.

*Stage 1: Quadratic Extension Multiplication*

Instead of multiplying field elements, we'll be multiplying elements in a "quadratic extension field". This is similar to complex numbers, but for fields. We'll now how two fields instead of one. 

    “fq_mul” is field multiplication
    “fq_add” is field addition
    “fq” is the field

Q. What is a field extension?
Field extension of a field F is another field F' which contains F. E.g. R (the field of real numbers) is a field extension of Q (the field of rational numbers). First start with our prime order field Fq where q is MNT4753.q. Then pick number in Fq which does not have a square root in Fq, e.g. 13. Now define the field called Fq[x]/(x^2 = 13). This is the field obtained by adding an "imaginary" square root x for 13 to Fq, similiar to how complex numbers are constructed from real numbers by adding an "imaginary" square root i for -1 to R. 

The elements of Fq[x]/(x^2 = 13) are sums of the form a_0 + a_1(x) where a_0 and a_1 are elements of Fq. This is a field extension of Fq since Fq is contained in this field as the elements with a_1 = 0. We call this field Fq^2 since it has q^2 elements. 

In code, you can think of an element of Fq^2 as a pair (a_0, a_1) where each of a_0 and a_1 is an element of Fq (e.g. imagine struct { a0 : Fq, a1 : Fq }). Addition and multiplication for Fq^2 is defined in the following manner:
```
    Addition: (a_0 + a_1x) + (b_0 + b_1x)
           = (a_0 + b_0) + (a_1 + b_1)x
    
    Multiplication: (a_0+ a_1x)(b_0 + b_1x) 
           = a_0b_0 + a_0b1x + b_0a1x + a_1b_1x^2 
​           = a_0b_0 + a_0b1x + b_0a1x + 13a_1b_1
           = (a_0b_0 + 13a_1b_1) + (a_0b_1 + b_0a_1)x
 ```

Psuedocode:
```
var alpha = fq(13);

var fq2_add = (a, b) => {
  return {
    a: fq_add(a.a0, b.a0),
    b: fq_add(a.a1, b.a1)
  };
};

var fq2_mul = (a, b) => {
  var a0_b0 = fq_mul(a.a0, b.a0);
  var a1_b1 = fq_mul(a.a1, b.a1);
  var a1_b0 = fq_mul(a.a1, b.a0);
  var a0_b1 = fq_mul(a.a0, b.a1);
  return {
    a0: fq_add(a0_b0, fq_mul(a1_b1, alpha)),
    a1: fq_add(a1_b0, a0_b1)
  };
};
```
*Stage 2: Cubic Extension Multiplication*

The elements of the cubic extension field Fq^3 are of the form: a0 + a1x + a2x^2. This is an extension of field Fq since it has x^3 elements, where each element is a tuple (a0, a1, a2). 

```
    Addition: (a_0 + a_1x + a_2x^2) + (b_0 + b_1x + b_2x^2)
           = (a_0 + b_0) + (a_1 + b_1)x + (a_1 + b_1)x^2
    
    Multiplication: (a_0 + a_1x + a_2x^2)(b_0 + b_1x + b_2x^2) 
           = a_0b_0 + a_0b1x + a_0b_2x^2 + a1b_0x + a_1b_1x^2 + a_1b_2x^3 + a_2b_0x^2 + a_2b_1x^3 + a_2b_2x^4
           = a_0b_0 + a_0b1x + a_0b_2x^2 + a1b_0x + a_1b_1x^2 + 11a_1b_2 + a_2b_0x^2 + 11a_2b_1 + 11a_2b_2x
           = (a_0b_0 + 11a_1b_2 + 11a_2b_1) + (a_0b1 + a1b_0 + 11a_2b_2)x + (a_0b_2 + a_1b_1 + a_2b_0)x^2

*Stage 3: Curve Operations*

Problem: Perform group operations for four elliptic curves.

A single SNARK proving / verifying system is a pair of elliptic curves (G1, G2). Since we're optimizing it for both MNT4 and MNT6, we have 4 curves:
  1. MNT4 G1
  2. MNT4 G2
  3. MNT6 G1
  4. MNT6 G2

Each curve is specified by a pair of two of these elements from the finite fields, specifically:
  MNT4 G1: (Fq, Fq)
  MNT4 G2: (Fq2, Fq2)
  MNT6 G1: (Fq, Fq)
  MNT6 G2: (Fq3, Fq3)

To refresh, G1 and G2 are cyclic groups of prime order q, with generator p. 