import numpy as np
import matplotlib.pyplot as plt

def xor_all(vals):
    out = 0
    for v in vals:
        out ^= int(v)
    return out

def rank_gf2(vectors, d):
    rows = [int(v) for v in vectors]
    rank = 0
    for col in range(d):
        pivot = None
        for i in range(rank, len(rows)):
            if (rows[i] >> col) & 1:
                pivot = i
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for i in range(len(rows)):
            if i != rank and ((rows[i] >> col) & 1):
                rows[i] ^= rows[rank]
        rank += 1
    return rank

def parity_matched_time(Delta):
    x = np.pi * Delta / 2
    candidates = [t for t in range(max(0, int(np.floor(x))-4),
                                   int(np.ceil(x))+5)
                  if (t-Delta) % 2 == 0]
    return min(candidates, key=lambda t: abs(t-x))

def hypercube_generators(d):
    return [1 << i for i in range(d)]

def augmented_cube_generators(n):
    gens = [1 << i for i in range(n)]
    gens += [(1 << i) - 1 for i in range(2, n+1)]
    return sorted(set(gens))

rng = np.random.default_rng(7)

def random_connected_generators(d, Delta):
    population = np.arange(1, 1 << d, dtype=int)
    for _ in range(10000):
        Omega = rng.choice(population, size=Delta, replace=False).tolist()
        if rank_gf2(Omega, d) == d:
            return Omega
    raise RuntimeError("Could not generate a connected cubelike graph.")

def exact_target_probability(d, Omega, T):
    N = 1 << d
    Delta = len(Omega)
    psi = np.zeros((Delta, N), dtype=np.complex128)
    psi[:, 0] = 1 / np.sqrt(Delta)
    positions = np.arange(N, dtype=int)

    for _ in range(T):
        mean = psi.mean(axis=0)
        psi = 2 * mean[None, :] - psi

        shifted = np.empty_like(psi)
        for j, omega in enumerate(Omega):
            shifted[j, positions ^ omega] = psi[j, positions]
        psi = shifted

    sigma = xor_all(Omega)
    return float(np.sum(np.abs(psi[:, sigma])**2).real), sigma

dims = list(range(4, 11))
families = {"Hypercube": [], "Augmented cube": [], "Random cubelike": []}

for d in dims:
    Omega = hypercube_generators(d)
    families["Hypercube"].append(
        exact_target_probability(d, Omega, parity_matched_time(len(Omega)))[0])

    Omega = augmented_cube_generators(d)
    families["Augmented cube"].append(
        exact_target_probability(d, Omega, parity_matched_time(len(Omega)))[0])

    Omega = random_connected_generators(d, 2*d - 1)
    families["Random cubelike"].append(
        exact_target_probability(d, Omega, parity_matched_time(len(Omega)))[0])

for label, probs in families.items():
    plt.plot(dims, probs, marker="o", label=label)

plt.xlabel("Dimension d")
plt.ylabel(r"Target probability $p_T(\sigma)$")
plt.title(r"Grover-coined cubelike walks at parity-matched $T \approx \pi\Delta/2$")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
