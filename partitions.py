import sys
import time

def build_pentagonal(n):
    """Precompute generalized pentagonal numbers and signs up to n."""
    pentagonal = []
    k = 1
    while True:
        g1 = k * (3 * k - 1) // 2
        g2 = k * (3 * k + 1) // 2
        if g1 > n:
            break
        sign = 1 if k % 2 == 1 else -1
        pentagonal.append((g1, sign))
        if g2 <= n:
            pentagonal.append((g2, sign))
        k += 1
    return pentagonal


def partition_mod(n, mod=10**9 + 7):
    """Compute p(0)..p(n) modulo mod using Euler's pentagonal theorem."""
    p = [0] * (n + 1)
    p[0] = 1
    pentagonal = build_pentagonal(n)
    for i in range(1, n + 1):
        total = 0
        for g, sign in pentagonal:
            if g > i:
                break
            total += sign * p[i - g]
        p[i] = total % mod
    return p


def partition_mod_compact(n, mod=10**9 + 7):
    """Memory-efficient version using array('I') — 4 bytes per entry."""
    from array import array
    p = array('I', [0]) * (n + 1)
    p[0] = 1
    pentagonal = build_pentagonal(n)
    for i in range(1, n + 1):
        total = 0
        for g, sign in pentagonal:
            if g > i:
                break
            total += sign * p[i - g]
        p[i] = total % mod
    return p


def partition_exact(n):
    """Compute exact p(0)..p(n) using Python big integers."""
    p = [0] * (n + 1)
    p[0] = 1
    pentagonal = build_pentagonal(n)
    for i in range(1, n + 1):
        total = 0
        for g, sign in pentagonal:
            if g > i:
                break
            total += sign * p[i - g]
        p[i] = total
    return p


def partition_numba(n, mod=10**9 + 7):
    """Fast version using Numba. Requires: pip install numba numpy"""
    import numpy as np
    from numba import njit

    @njit(cache=True)
    def _run(n, mod):
        p = np.zeros(n + 1, dtype=np.int64)
        p[0] = 1
        max_k = 1
        while max_k * (3 * max_k - 1) // 2 <= n:
            max_k += 1
        max_k += 1
        g = np.empty(2 * max_k, dtype=np.int64)
        s = np.empty(2 * max_k, dtype=np.int64)
        idx = 0
        k = 1
        while True:
            g1 = k * (3 * k - 1) // 2
            g2 = k * (3 * k + 1) // 2
            if g1 > n:
                break
            sign = 1 if k % 2 == 1 else -1
            g[idx] = g1; s[idx] = sign; idx += 1
            if g2 <= n:
                g[idx] = g2; s[idx] = sign; idx += 1
            k += 1
        for i in range(1, n + 1):
            total = 0
            for j in range(idx):
                if g[j] > i:
                    break
                total += s[j] * p[i - g[j]]
            p[i] = total % mod
        return p

    return _run(n, mod)


def ask_int(prompt, default, minimum=0):
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        print(f"  Invalid integer — using default {default}.")
        return default
    if val < minimum:
        print(f"  Must be >= {minimum} — using default {default}.")
        return default
    return val


def main():
    print("=" * 60)
    print("  Partition function p(n) calculator")
    print("=" * 60)
    print()
    print("Choose a mode:")
    print("  1) Modular  — p(n) mod M   (fast, works up to 10 million+)")
    print("  2) Compact  — modular, low-memory variant (array-based)")
    print("  3) Exact    — full big-integer values (small n only, <10000)")
    print("  4) Numba    — modular, fastest (needs numba + numpy)")
    print()

    mode = input("Mode [1-4, default 1]: ").strip() or "1"
    if mode not in {"1", "2", "3", "4"}:
        print(f"Unknown mode '{mode}' — defaulting to 1.")
        mode = "1"

    n = ask_int("Compute up to n =", 1_000_000, minimum=0)

    if mode in {"1", "2", "4"}:
        mod = ask_int("Modulus M (default 10^9+7)", 10**9 + 7, minimum=1)
    else:
        mod = None

    print()
    print(f"Running mode {mode} for n = {n:,}"
          + (f", mod = {mod:,}" if mod else " (exact big integers)"))
    print("-" * 60)

    start = time.time()
    try:
        if mode == "1":
            p = partition_mod(n, mod)
        elif mode == "2":
            p = partition_mod_compact(n, mod)
        elif mode == "3":
            p = partition_exact(n)
        else:  # mode 4
            p = partition_numba(n, mod)
    except ImportError as e:
        print(f"Import error: {e}")
        print("Install the dependency (e.g. `pip install numba numpy`) and retry.")
        return
    except MemoryError:
        print("Out of memory — try mode 2 (compact) or a smaller n.")
        return

    elapsed = time.time() - start

    print(f"Done in {elapsed:.2f} s")
    print()
    print(f"p({n:,}) = {p[n]}")

    # Show a few extra checkpoints if user wants them
    if n >= 100:
        print()
        print("Reference checkpoints:")
        checks = [(100, 190569292), (1000, None)]
        for k, expected in checks:
            if k <= n:
                val = p[k]
                if expected is not None and mod is None:
                    ok = "OK" if val == expected else f"expected {expected}"
                    print(f"  p({k}) = {val}  ({ok})")
                else:
                    print(f"  p({k}) = {val}")

    # Offer to print a range of values
    print()
    show = input("Print a range of p(i) values? [y/N]: ").strip().lower()
    if show == "y":
        lo = ask_int("  start index i =", max(0, n - 10), minimum=0)
        hi = ask_int("  end index   i =", n, minimum=0)
        lo, hi = min(lo, hi), max(lo, hi)
        hi = min(hi, n)
        for i in range(lo, hi + 1):
            print(f"  p({i}) = {p[i]}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted.")
        sys.exit(1)