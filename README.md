# Misalignment Detector

**Provably Non-Gameable Alignment Auditing via Octonionic Algebra**

## What This Does

Uses the octonionic associator to detect when an AI agent (or any strategic actor) misrepresents its objectives. Unlike heuristic alignment checks, this is *mathematically proven* — based on Theorem 26.12 from non-associative algebra.

## The Core Insight

In associative math (the kind everything else uses), grouping doesn't matter: `(ab)c = a(bc)`. An agent can swap its representation without any detectable signature change. Alignment checks built on associative algebra are **inherently gameable**.

The octonions are the largest normed division algebra, and they are *non-associative*: `(ab)c ≠ a(bc)`. The difference — the **associator** `[a,b,c] = (ab)c - a(bc)` — is generically nonzero. When you encode agent objectives as octonions and compute the associator signature over all agent triples, **any change to any agent's representation is detectable**.

This is not a heuristic. It is a theorem.

## Results

| Setting | Gaming Attempts | Detected | Detection Rate |
|---------|----------------|----------|----------------|
| **Octonionic** (non-associative) | 300 | 300 | **100%** |
| Quaternionic (associative) | 300 | 0 | 0% |

Targeted gaming (agent strategically tries to improve alignment score without changing signature): **0 successful attempts out of 300.**

## Quick Start

```bash
pip install numpy
python demo_alignment.py
```

Runs 8 demonstrations: tree enumeration, associator proof, gaming detection, targeted gaming analysis, bracketing sensitivity, AI alignment audit, team optimization, and scaling tests.

## Files

- `octonion.py` — Octonion algebra (Fano plane multiplication, associator, 9 self-tests)
- `alignment.py` — Non-gameable alignment engine (tree-averaged alignment, gaming detection, AI audit)
- `demo_alignment.py` — Full demonstration with 8 sections

## Applications

- **AI alignment auditing** — detect misrepresented objectives in multi-agent systems
- **Team composition** — gaming-resistant agent selection
- **Governance & voting** — non-gameable policy alignment verification
- **Mechanism design** — collusion-resistant market protocols

## Based On

Chapter 26 of *The Octonionic Rewrite* — "Non-Gameable Alignment Functions." Theorem 26.12 proves that alignment functions over the octonions are non-gameable: the associator makes every misrepresentation detectable.

## License

MIT — see [LICENSE](LICENSE)
