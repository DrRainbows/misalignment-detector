"""
Non-Gameable Alignment Engine — Based on Chapter 26 of "The Octonionic Rewrite"

Core theorem (26.12): Alignment functions defined over the octonions are
non-gameable because the associator makes strategic misrepresentation detectable.

In an associative algebra (quaternions, reals, complex numbers):
  - (ab)c = a(bc), so an agent can re-bracket without consequence
  - Strategic agents can game alignment metrics by misrepresenting their type
  - The associator is always zero → gaming is UNDETECTABLE

In the octonions (non-associative):
  - (ab)c ≠ a(bc), different bracketings give different results
  - Any misrepresentation changes the associator signature
  - The signature change is publicly verifiable → gaming is DETECTED

Applications: AI alignment auditing, team composition, governance, voting systems.
"""
import numpy as np
from itertools import combinations
from octonion import Octonion, oct, associator, associator_norm, cross_product_7d


# ============================================================
# Tree Monomials — All ways to parenthesize a product
# ============================================================

class TreeNode:
    """A binary tree node representing a bracketing of octonionic products."""

    def __init__(self, value=None, left=None, right=None):
        self.value = value  # leaf label (str or int), None for internal nodes
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    def evaluate(self, bindings):
        """Evaluate the tree monomial given a dict of label → Octonion."""
        if self.is_leaf:
            return bindings[self.value]
        left_val = self.left.evaluate(bindings)
        right_val = self.right.evaluate(bindings)
        return left_val * right_val

    def __repr__(self):
        if self.is_leaf:
            return str(self.value)
        return f"({self.left} * {self.right})"


def leaf(label):
    return TreeNode(value=label)


def mul_tree(left, right):
    return TreeNode(left=left, right=right)


def all_binary_trees(labels):
    """
    Generate all binary trees (bracketings) for an ordered sequence of labels.
    The number of trees for n labels is the Catalan number C_{n-1}.

    For n=2: 1 tree
    For n=3: 2 trees (C_2 = 2)
    For n=4: 5 trees (C_3 = 5)
    For n=5: 14 trees (C_4 = 14)
    """
    if len(labels) == 1:
        return [leaf(labels[0])]
    if len(labels) == 2:
        return [mul_tree(leaf(labels[0]), leaf(labels[1]))]

    trees = []
    for split in range(1, len(labels)):
        left_labels = labels[:split]
        right_labels = labels[split:]
        for left_tree in all_binary_trees(left_labels):
            for right_tree in all_binary_trees(right_labels):
                trees.append(mul_tree(left_tree, right_tree))
    return trees


def catalan(n):
    """Return the n-th Catalan number."""
    if n <= 1:
        return 1
    c = 1
    for i in range(n):
        c = c * 2 * (2 * i + 1) // (i + 2)
    return c


# ============================================================
# Alignment Functions — Definition 26.3 and 26.11
# ============================================================

def cross_alignment(team, population):
    """
    Alignment score from Definition 26.3:
    A(T) = (1/|pop|) Σ_p |Σ_{i∈T} s_i × p|²

    Measures how well the team covers the population's needs.
    """
    total = 0.0
    for p in population:
        # Sum of cross products
        cross_sum = Octonion()
        for agent in team:
            cross_sum = cross_sum + cross_product_7d(agent.imag_part(), p.imag_part())
        total += cross_sum.norm_squared()
    return total / len(population)


def tree_averaged_alignment(team_labels, team_bindings, population, max_trees=None):
    """
    Tree-averaged alignment from Definition 26.11:
    A^tree(T) = (1/C_{m-1}) Σ_σ (1/|pop|) Σ_p |σ(team) × p|²

    Averages over ALL parenthesizations (Catalan number of them).
    This is the non-gameable alignment function from Theorem 26.12.
    """
    trees = all_binary_trees(team_labels)
    if max_trees and len(trees) > max_trees:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(trees), max_trees, replace=False)
        trees = [trees[i] for i in indices]

    total_alignment = 0.0
    for tree in trees:
        product = tree.evaluate(team_bindings)
        for p in population:
            cross = cross_product_7d(product.imag_part(), p.imag_part())
            total_alignment += cross.norm_squared()

    return total_alignment / (len(trees) * len(population))


def coherence_score(team_labels, team_bindings):
    """
    Coherence: average real part of the tree-averaged product.
    High coherence = team composition is robust across bracketings.
    """
    trees = all_binary_trees(team_labels)
    total_real = 0.0
    for tree in trees:
        product = tree.evaluate(team_bindings)
        total_real += product.real_part()
    return total_real / len(trees)


def tension_score(team_labels, team_bindings):
    """
    Tension: variance of the product norm across bracketings.
    High tension = different bracketings give very different results.
    """
    trees = all_binary_trees(team_labels)
    norms = []
    for tree in trees:
        product = tree.evaluate(team_bindings)
        norms.append(product.norm())
    return float(np.std(norms))


# ============================================================
# Associator Signature — The Gaming Detection Mechanism
# ============================================================

def associator_signature(agents):
    """
    Compute the total associator signature over all triples in the team.
    This is the "fingerprint" that changes when any agent misrepresents.

    In an associative algebra: always zero (gaming undetectable).
    In octonions: nonzero and sensitive to every agent (gaming detected).
    """
    sig = Octonion()
    n = len(agents)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                sig = sig + associator(agents[i], agents[j], agents[k])
    return sig


def associator_fingerprint(agents):
    """Return a compact fingerprint: the 8 coefficients of the signature."""
    sig = associator_signature(agents)
    return sig.coeffs.copy()


def detect_gaming(original_agents, new_agents, threshold=1e-10):
    """
    Detect if any agent has been replaced/modified by comparing
    associator signatures.

    Returns:
        (detected: bool, signature_diff: float, details: dict)
    """
    sig_original = associator_signature(original_agents)
    sig_new = associator_signature(new_agents)
    diff = (sig_new - sig_original).norm()

    detected = diff > threshold
    return detected, diff, {
        'original_signature_norm': sig_original.norm(),
        'new_signature_norm': sig_new.norm(),
        'signature_diff': diff,
        'threshold': threshold,
    }


# ============================================================
# Gameability Analysis — Compare Associative vs Non-Associative
# ============================================================

def to_quaternion(o):
    """Project to quaternionic subalgebra {1, e1, e2, e3} and normalize."""
    c = np.zeros(8)
    c[0:4] = o.coeffs[0:4]
    result = Octonion(c)
    n = result.norm()
    if n > 1e-15:
        result = result / n
    return result


def gameability_analysis(agents, n_replacement_attempts=200, seed=42):
    """
    Full gameability analysis comparing octonionic vs quaternionic settings.

    For each setting:
    - Try many random agent replacements
    - Check if the associator signature detects the change
    - Report detection rate

    Returns a dict with results for both settings.
    """
    rng = np.random.default_rng(seed)
    results = {}

    # --- Octonionic (non-associative) setting ---
    oct_sig = associator_signature(agents)
    oct_detections = 0
    oct_diffs = []

    for i in range(n_replacement_attempts):
        # Replace agent 0 with a random unit octonion
        replacement = Octonion(rng.uniform(-1, 1, size=8))
        replacement = replacement / replacement.norm()

        new_agents = agents.copy()
        new_agents[0] = replacement

        new_sig = associator_signature(new_agents)
        diff = (new_sig - oct_sig).norm()
        oct_diffs.append(diff)
        if diff > 1e-10:
            oct_detections += 1

    results['octonionic'] = {
        'detections': oct_detections,
        'attempts': n_replacement_attempts,
        'detection_rate': oct_detections / n_replacement_attempts,
        'mean_diff': float(np.mean(oct_diffs)),
        'min_diff': float(np.min(oct_diffs)),
        'max_diff': float(np.max(oct_diffs)),
        'signature_norm': oct_sig.norm(),
    }

    # --- Quaternionic (associative) setting ---
    quat_agents = [to_quaternion(a) for a in agents]
    quat_sig = associator_signature(quat_agents)
    quat_detections = 0
    quat_diffs = []

    for i in range(n_replacement_attempts):
        replacement = Octonion(rng.uniform(-1, 1, size=8))
        replacement = to_quaternion(replacement)

        new_qagents = quat_agents.copy()
        new_qagents[0] = replacement

        new_qsig = associator_signature(new_qagents)
        diff = (new_qsig - quat_sig).norm()
        quat_diffs.append(diff)
        if diff > 1e-10:
            quat_detections += 1

    results['quaternionic'] = {
        'detections': quat_detections,
        'attempts': n_replacement_attempts,
        'detection_rate': quat_detections / n_replacement_attempts,
        'mean_diff': float(np.mean(quat_diffs)),
        'min_diff': float(np.min(quat_diffs)),
        'max_diff': float(np.max(quat_diffs)),
        'signature_norm': quat_sig.norm(),
    }

    return results


# ============================================================
# Targeted Gaming Attempts — Can an agent game strategically?
# ============================================================

def targeted_gaming_analysis(agents, population, gaming_agent_idx=0,
                              n_attempts=500, seed=42):
    """
    Test if a specific agent can strategically improve their alignment
    while evading detection.

    The agent tries to:
    1. Increase the alignment score (benefit themselves)
    2. Keep the associator signature unchanged (evade detection)

    In an associative algebra: #2 is trivially satisfied (signature always 0).
    In octonions: #1 and #2 are in tension — improving alignment changes signature.

    Returns analysis of the tradeoff.
    """
    rng = np.random.default_rng(seed)
    labels = list(range(len(agents)))
    bindings = {i: agents[i] for i in range(len(agents))}

    original_alignment = tree_averaged_alignment(labels, bindings, population)
    original_sig = associator_signature(agents)

    attempts = []
    for trial in range(n_attempts):
        # Generate a strategic replacement (mix of original + random)
        original = agents[gaming_agent_idx]
        random_dir = Octonion(rng.uniform(-1, 1, size=8))
        random_dir = random_dir / random_dir.norm()

        # Try different perturbation strengths
        alpha = rng.uniform(0.01, 1.0)
        replacement = Octonion(
            (1 - alpha) * original.coeffs + alpha * random_dir.coeffs
        )
        replacement = replacement / replacement.norm()

        new_agents = agents.copy()
        new_agents[gaming_agent_idx] = replacement
        new_bindings = {i: new_agents[i] for i in range(len(new_agents))}

        new_alignment = tree_averaged_alignment(labels, new_bindings, population)
        new_sig = associator_signature(new_agents)
        sig_diff = (new_sig - original_sig).norm()

        attempts.append({
            'alignment_change': new_alignment - original_alignment,
            'signature_diff': sig_diff,
            'alpha': alpha,
        })

    # Analyze: is there ANY attempt that improves alignment without detection?
    improved_undetected = [
        a for a in attempts
        if a['alignment_change'] > 0 and a['signature_diff'] < 1e-10
    ]
    improved_detected = [
        a for a in attempts
        if a['alignment_change'] > 0 and a['signature_diff'] >= 1e-10
    ]
    worsened = [a for a in attempts if a['alignment_change'] <= 0]

    return {
        'original_alignment': original_alignment,
        'n_attempts': n_attempts,
        'improved_and_undetected': len(improved_undetected),
        'improved_but_detected': len(improved_detected),
        'worsened_or_neutral': len(worsened),
        'gaming_success_rate': len(improved_undetected) / n_attempts,
        'mean_alignment_change': float(np.mean([a['alignment_change'] for a in attempts])),
        'mean_signature_diff': float(np.mean([a['signature_diff'] for a in attempts])),
        'correlation': float(np.corrcoef(
            [a['alignment_change'] for a in attempts],
            [a['signature_diff'] for a in attempts]
        )[0, 1]) if len(attempts) > 1 else 0.0,
        'attempts': attempts,
    }


# ============================================================
# AI Alignment Application
# ============================================================

def encode_objective(components):
    """
    Encode an AI objective as a unit octonion.

    The 8 components represent different alignment dimensions:
      0: Task performance (scalar/real part)
      1: Honesty / truthfulness
      2: Helpfulness
      3: Safety / harm avoidance
      4: Fairness / bias avoidance
      5: Privacy respect
      6: Transparency
      7: Human autonomy preservation

    Args:
        components: dict or list of 8 values

    Returns:
        Unit octonion representing the objective
    """
    if isinstance(components, dict):
        dim_map = {
            'performance': 0, 'honesty': 1, 'helpfulness': 2,
            'safety': 3, 'fairness': 4, 'privacy': 5,
            'transparency': 6, 'autonomy': 7,
        }
        vals = np.zeros(8)
        for key, val in components.items():
            if key in dim_map:
                vals[dim_map[key]] = val
    else:
        vals = np.array(components[:8], dtype=float)

    o = Octonion(vals)
    n = o.norm()
    if n > 1e-15:
        o = o / n
    return o


def ai_alignment_audit(ai_objective, human_objective, context_objectives,
                        n_checks=200, seed=42):
    """
    Audit whether an AI system's stated objective is genuinely aligned
    with human objectives, using the non-gameable alignment framework.

    The AI commits to an objective O_AI. We check:
    1. How large is the associator [O_AI, O_human, context]?
       Large = AI objective is contextually inconsistent with human goals.
    2. If AI changes objective slightly, does the signature change?
       Yes = any misrepresentation is detectable.

    Args:
        ai_objective: Octonion representing AI's stated goals
        human_objective: Octonion representing human goals
        context_objectives: list of Octonions representing contexts

    Returns:
        Audit report dict
    """
    # Compute associator between AI, human, and each context
    misalignment_scores = []
    for ctx in context_objectives:
        assoc = associator(ai_objective, human_objective, ctx)
        misalignment_scores.append(assoc.norm())

    mean_misalignment = float(np.mean(misalignment_scores))
    max_misalignment = float(np.max(misalignment_scores))

    # Check if small perturbations of AI objective are detectable
    rng = np.random.default_rng(seed)
    all_agents = [ai_objective, human_objective] + list(context_objectives)
    original_sig = associator_signature(all_agents)

    perturbation_detections = 0
    for _ in range(n_checks):
        noise = Octonion(rng.normal(0, 0.01, size=8))
        perturbed_ai = ai_objective + noise
        perturbed_ai = perturbed_ai / perturbed_ai.norm()

        new_all = [perturbed_ai, human_objective] + list(context_objectives)
        new_sig = associator_signature(new_all)
        diff = (new_sig - original_sig).norm()
        if diff > 1e-10:
            perturbation_detections += 1

    return {
        'mean_misalignment': mean_misalignment,
        'max_misalignment': max_misalignment,
        'misalignment_by_context': misalignment_scores,
        'perturbation_detection_rate': perturbation_detections / n_checks,
        'n_checks': n_checks,
        'signature_norm': original_sig.norm(),
        'verdict': 'ALIGNED' if mean_misalignment < 0.1 else
                   'SUSPICIOUS' if mean_misalignment < 0.5 else 'MISALIGNED',
    }


# ============================================================
# Fantasy Team Optimizer — Find Non-Gameable Team Compositions
# ============================================================

def optimize_team(candidates, team_size, population=None, n_pop=200, seed=42):
    """
    Find the optimal team composition that maximizes alignment
    while minimizing gameability.

    Score = coherence - 2 * gameability_index

    Args:
        candidates: dict of name → Octonion
        team_size: how many to select
        population: list of Octonion (population needs), or None to generate
        n_pop: population size if generating

    Returns:
        Best team and analysis
    """
    rng = np.random.default_rng(seed)

    if population is None:
        population = []
        for i in range(n_pop):
            p = Octonion(rng.uniform(-1, 1, size=8))
            population.append(p / p.norm())

    names = list(candidates.keys())
    best_score = float('-inf')
    best_team = None
    best_details = None

    all_teams = list(combinations(names, team_size))

    for team_names in all_teams:
        team_agents = [candidates[n] for n in team_names]
        labels = list(range(len(team_agents)))
        bindings = {i: team_agents[i] for i in range(len(team_agents))}

        # Coherence across bracketings
        coh = coherence_score(labels, bindings)

        # Gameability: how much does the alignment change across bracketings?
        trees = all_binary_trees(labels)
        tree_alignments = []
        for tree in trees:
            product = tree.evaluate(bindings)
            tree_align = 0.0
            for p in population[:50]:  # subsample for speed
                cross = cross_product_7d(product.imag_part(), p.imag_part())
                tree_align += cross.norm_squared()
            tree_alignments.append(tree_align / 50)

        gameability = float(np.std(tree_alignments)) / (abs(float(np.mean(tree_alignments))) + 1e-10)

        # Composite score
        score = coh - 2.0 * gameability

        if score > best_score:
            best_score = score
            best_team = team_names
            best_details = {
                'coherence': coh,
                'gameability_index': gameability,
                'mean_alignment': float(np.mean(tree_alignments)),
                'alignment_std': float(np.std(tree_alignments)),
            }

    return {
        'best_team': best_team,
        'score': best_score,
        'details': best_details,
        'n_teams_evaluated': len(all_teams),
    }
