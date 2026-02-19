"""
Octonion algebra implementation using the Fano plane.
Basis: {1, e1, e2, e3, e4, e5, e6, e7}
An octonion is represented as 8 real numbers (x0, x1, ..., x7)
where x = x0 + x1*e1 + x2*e2 + ... + x7*e7.

From "The Octonionic Rewrite" Appendix C, verified against the book's test suite.
"""
import numpy as np

# Fano plane triples: (i, j, k) means e_i * e_j = +e_k
# All cyclic permutations also hold: e_j * e_k = +e_i, e_k * e_i = +e_j
FANO_TRIPLES = [
    (1, 2, 3),
    (1, 4, 5),
    (2, 4, 6),
    (3, 4, 7),
    (1, 7, 6),
    (2, 5, 7),
    (3, 6, 5),
]


def _build_multiplication_table():
    """
    Build the 8x8 multiplication table for octonion basis elements.
    table[i][j] = (sign, index) means e_i * e_j = sign * e_index.
    Index 0 represents the real unit 1.
    """
    table = [[(0, 0)] * 8 for _ in range(8)]

    # e_0 * e_0 = 1
    table[0][0] = (1, 0)

    # e_0 * e_i = e_i and e_i * e_0 = e_i
    for i in range(1, 8):
        table[0][i] = (1, i)
        table[i][0] = (1, i)

    # e_i * e_i = -1 for i >= 1
    for i in range(1, 8):
        table[i][i] = (-1, 0)

    # Fano plane triples
    for (i, j, k) in FANO_TRIPLES:
        # Cyclic: e_i * e_j = +e_k
        table[i][j] = (1, k)
        table[j][k] = (1, i)
        table[k][i] = (1, j)
        # Anti-cyclic: e_j * e_i = -e_k
        table[j][i] = (-1, k)
        table[k][j] = (-1, i)
        table[i][k] = (-1, j)

    return table


MULT_TABLE = _build_multiplication_table()


class Octonion:
    """An element of the octonion algebra O."""

    __slots__ = ['coeffs']

    def __init__(self, coeffs=None):
        if coeffs is not None:
            self.coeffs = np.asarray(coeffs, dtype=float)
        else:
            self.coeffs = np.zeros(8)

    @classmethod
    def basis(cls, i):
        c = np.zeros(8)
        c[i] = 1.0
        return cls(c)

    @classmethod
    def real(cls, r):
        return cls([r, 0, 0, 0, 0, 0, 0, 0])

    @classmethod
    def random(cls, seed=None):
        rng = np.random.default_rng(seed)
        return cls(rng.uniform(-1, 1, size=8))

    def __repr__(self):
        parts = []
        names = ['1', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7']
        for i, (c, n) in enumerate(zip(self.coeffs, names)):
            if abs(c) < 1e-12:
                continue
            if i == 0:
                parts.append(f"{c:.6g}")
            else:
                parts.append(f"{c:+.6g}*{n}")
        return ' '.join(parts) if parts else '0'

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Octonion.real(other)
        return Octonion(self.coeffs + other.coeffs)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Octonion.real(other)
        return Octonion(self.coeffs - other.coeffs)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Octonion.real(other)
        return Octonion(other.coeffs - self.coeffs)

    def __neg__(self):
        return Octonion(-self.coeffs)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Octonion(self.coeffs * other)
        result = np.zeros(8)
        for i in range(8):
            ai = self.coeffs[i]
            if abs(ai) < 1e-15:
                continue
            for j in range(8):
                bj = other.coeffs[j]
                if abs(bj) < 1e-15:
                    continue
                sign, idx = MULT_TABLE[i][j]
                result[idx] += sign * ai * bj
        return Octonion(result)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Octonion(self.coeffs * other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Octonion(self.coeffs / other)
        return self * other.inverse()

    def conjugate(self):
        c = self.coeffs.copy()
        c[1:] = -c[1:]
        return Octonion(c)

    def norm_squared(self):
        return float(np.dot(self.coeffs, self.coeffs))

    def norm(self):
        return float(np.sqrt(self.norm_squared()))

    def inverse(self):
        ns = self.norm_squared()
        if ns < 1e-30:
            raise ZeroDivisionError("Cannot invert zero octonion")
        return self.conjugate() / ns

    def real_part(self):
        return float(self.coeffs[0])

    def imag_part(self):
        c = self.coeffs.copy()
        c[0] = 0.0
        return Octonion(c)

    def imag_vector(self):
        return self.coeffs[1:].copy()

    def dot(self, other):
        return float(np.dot(self.coeffs, other.coeffs))

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            other = Octonion.real(other)
        return np.allclose(self.coeffs, other.coeffs, atol=1e-10)

    def is_zero(self, tol=1e-10):
        return self.norm() < tol


# --- Convenience ---
def oct(x0=0, x1=0, x2=0, x3=0, x4=0, x5=0, x6=0, x7=0):
    return Octonion([x0, x1, x2, x3, x4, x5, x6, x7])


e0 = Octonion.basis(0)
e1 = Octonion.basis(1)
e2 = Octonion.basis(2)
e3 = Octonion.basis(3)
e4 = Octonion.basis(4)
e5 = Octonion.basis(5)
e6 = Octonion.basis(6)
e7 = Octonion.basis(7)


# --- Core functions ---
def associator(a, b, c):
    """[a, b, c] = (a*b)*c - a*(b*c)"""
    return (a * b) * c - a * (b * c)


def associator_norm(a, b, c):
    return associator(a, b, c).norm()


def cross_product_7d(a, b):
    """7D cross product for imaginary octonions: Im(a*b)"""
    return (a * b).imag_part()


def commutator(a, b):
    """[a, b] = a*b - b*a"""
    return a * b - b * a


def commutator_norm(a, b):
    return commutator(a, b).norm()


# --- Self-test ---
def run_tests():
    print("Running octonion algebra tests...")

    # Test 1: Basis element squares
    for i in range(1, 8):
        ei = Octonion.basis(i)
        assert (ei * ei) == Octonion.real(-1), f"e{i}^2 should be -1"
    print("  PASS: All e_i^2 = -1")

    # Test 2: Fano plane products
    for (i, j, k) in FANO_TRIPLES:
        ei, ej, ek = Octonion.basis(i), Octonion.basis(j), Octonion.basis(k)
        assert (ei * ej) == ek, f"e{i}*e{j} should be e{k}"
        assert (ej * ei) == -ek, f"e{j}*e{i} should be -e{k}"
        assert (ej * ek) == ei, f"e{j}*e{k} should be e{i}"
        assert (ek * ei) == ej, f"e{k}*e{i} should be e{j}"
    print("  PASS: Fano plane products correct")

    # Test 3: Norm multiplicativity
    a = oct(1, 2, 3, 4, 5, 6, 7, 8)
    b = oct(-3, 1, 4, 1, 5, 9, 2, 6)
    ab = a * b
    assert abs(ab.norm() - a.norm() * b.norm()) < 1e-10
    print(f"  PASS: |a*b| = |a|*|b| (norm multiplicativity)")

    # Test 4: Inverse
    a = oct(1, 2, -1, 3, 0, 4, -2, 1)
    a_inv = a.inverse()
    product = a * a_inv
    assert product == Octonion.real(1)
    print("  PASS: a * a^{-1} = 1")

    # Test 5: Conjugation anti-homomorphism
    a = oct(1, 2, 3, 4, 5, 6, 7, 8)
    b = oct(-1, 3, -2, 5, 0, 1, -4, 2)
    assert (a * b).conjugate() == b.conjugate() * a.conjugate()
    print("  PASS: conj(a*b) = conj(b)*conj(a)")

    # Test 6: Alternativity
    a = oct(1, 2, 3, 4, 5, 6, 7, 8)
    b = oct(-1, 3, -2, 5, 0, 1, -4, 2)
    assert (a * a) * b == a * (a * b)
    assert (b * a) * a == b * (a * a)
    print("  PASS: Alternativity holds")

    # Test 7: Non-associativity
    result = associator(e1, e2, e4)
    assert not result.is_zero()
    expected = Octonion([0, 0, 0, 0, 0, 0, 0, 2])  # 2*e7
    assert result == expected, f"Expected 2*e7, got {result}"
    print("  PASS: [e1, e2, e4] = 2*e7 (non-associativity confirmed)")

    # Test 8: Associator vanishes on Fano triples
    assert associator(e1, e2, e3).is_zero()
    print("  PASS: [e1, e2, e3] = 0 (quaternionic subalgebra)")

    # Test 9: Associator antisymmetry
    a, b, c = oct(0, 1, 2, 0, 0, 0, 0, 0), oct(0, 0, 0, 1, 1, 0, 0, 0), oct(0, 0, 0, 0, 0, 1, 0, 1)
    abc = associator(a, b, c)
    assert (abc + associator(b, a, c)).is_zero()
    assert (abc + associator(a, c, b)).is_zero()
    print("  PASS: Associator is antisymmetric")

    print("All tests passed!\n")


if __name__ == '__main__':
    run_tests()
