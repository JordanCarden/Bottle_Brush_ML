from __future__ import annotations

import ast
import math
import random
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


_LABEL_RE = re.compile(r"^([ES])(\d+)$")


@dataclass(frozen=True, slots=True)
class Token:
    position: int
    chem: str
    length: int

    @property
    def label(self) -> str:
        return f"{self.chem}{self.length}"


def parse_architecture(text: str) -> List[Token]:
    s = (text or "").strip()
    if (s.startswith("\"") and s.endswith("\"")) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    parsed = ast.literal_eval(s)
    if isinstance(parsed, str):
        parsed = ast.literal_eval(parsed)

    tokens: List[Token] = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError(f"Invalid token entry: {item!r}")
        position = int(item[0])
        label = str(item[1]).strip()
        match = _LABEL_RE.match(label)
        if not match:
            raise ValueError(f"Invalid label: {label!r}")
        chem, length_str = match.groups()
        length = int(length_str)
        tokens.append(Token(position=position, chem=chem, length=length))

    tokens.sort(key=lambda tok: tok.position)
    positions = [tok.position for tok in tokens]
    if len(set(positions)) != len(positions):
        raise ValueError("Duplicate backbone positions found in architecture.")
    expected = list(range(1, len(tokens) + 1))
    if positions != expected:
        raise ValueError(
            f"Backbone positions must be consecutive starting at 1 (got {positions[:5]}...)."
        )
    return tokens


def serialize_architecture(tokens: Sequence[Token]) -> str:
    items = [(tok.position, tok.label) for tok in sorted(tokens, key=lambda t: t.position)]
    return repr(items)


def signed_lengths(tokens: Sequence[Token]) -> List[int]:
    values: List[int] = []
    for tok in tokens:
        length = int(tok.length)
        if tok.chem == "E":
            values.append(-length)
        else:
            values.append(length)
    return values


def blockiness(tokens: Sequence[Token]) -> float:
    signs: List[int] = []
    for value in signed_lengths(tokens):
        if value > 0:
            signs.append(1)
        elif value < 0:
            signs.append(-1)
        else:
            signs.append(0)
    filtered = [s for s in signs if s != 0]
    if len(filtered) < 2:
        return float("nan")
    same_pairs = sum(1 for idx in range(1, len(filtered)) if filtered[idx] == filtered[idx - 1])
    return same_pairs / (len(filtered) - 1)


def gini_coefficient(values: Iterable[int]) -> float:
    arr = [abs(int(v)) for v in values if int(v) != 0]
    if len(arr) < 2:
        return float("nan")
    total = sum(arr)
    if total == 0:
        return float("nan")
    arr_sorted = sorted(arr)
    n = len(arr_sorted)
    weighted_sum = sum((idx + 1) * v for idx, v in enumerate(arr_sorted))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def gini_coefficient_all_sites(values: Iterable[int]) -> float:
    """Gini coefficient over absolute values, including zeros.

    This treats ungrafted sites (0 length) as valid positions and will therefore
    reflect both grafting density and grafted-length inequality.
    """

    arr = [abs(int(v)) for v in values]
    if len(arr) < 2:
        return 0.0
    total = sum(arr)
    if total == 0:
        return 0.0
    arr_sorted = sorted(arr)
    n = len(arr_sorted)
    weighted_sum = sum((idx + 1) * v for idx, v in enumerate(arr_sorted))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def coefficient_of_variation(values: Iterable[int]) -> float:
    arr = [abs(int(v)) for v in values if int(v) != 0]
    if len(arr) < 2:
        return float("nan")
    mean = sum(arr) / len(arr)
    if math.isclose(mean, 0.0, abs_tol=1e-12):
        return float("nan")
    variance = sum((x - mean) ** 2 for x in arr) / len(arr)
    return math.sqrt(variance) / mean


def length_weighted_peo_fraction(tokens: Sequence[Token]) -> float:
    sum_e = sum(tok.length for tok in tokens if tok.chem == "E" and tok.length > 0)
    sum_s = sum(tok.length for tok in tokens if tok.chem == "S" and tok.length > 0)
    total = sum_e + sum_s
    return (sum_e / total) if total else 0.0


def total_length(tokens: Sequence[Token], chem: str | None = None) -> int:
    return sum(
        tok.length for tok in tokens if tok.length > 0 and (chem is None or tok.chem == chem)
    )


def n_transitions(tokens: Sequence[Token]) -> int:
    nonzero = [tok.chem for tok in tokens if tok.length > 0]
    return sum(1 for idx in range(1, len(nonzero)) if nonzero[idx] != nonzero[idx - 1])


def end_loading(tokens: Sequence[Token], k: int = 3, chem: str | None = None) -> float:
    n = len(tokens)
    if n == 0:
        return float("nan")
    total = total_length(tokens, chem=chem)
    if total <= 0:
        return float("nan")

    end_positions = set(range(1, min(k, n) + 1)) | set(range(max(1, n - k + 1), n + 1))
    end_total = sum(
        tok.length
        for tok in tokens
        if tok.length > 0
        and tok.position in end_positions
        and (chem is None or tok.chem == chem)
    )
    return end_total / total


def length_weighted_centroid(tokens: Sequence[Token], chem: str) -> float:
    mass = sum(tok.length for tok in tokens if tok.chem == chem and tok.length > 0)
    if mass <= 0:
        return float("nan")
    return (
        sum(tok.position * tok.length for tok in tokens if tok.chem == chem and tok.length > 0)
        / mass
    )


def centroid_separation(tokens: Sequence[Token], normalize: bool = True) -> float:
    c_e = length_weighted_centroid(tokens, "E")
    c_s = length_weighted_centroid(tokens, "S")
    if math.isnan(c_e) or math.isnan(c_s):
        return float("nan")
    distance = abs(c_e - c_s)
    return distance / len(tokens) if normalize and len(tokens) else distance


def sigma_grafting(tokens: Sequence[Token]) -> float:
    if not tokens:
        return 0.0
    grafted = sum(1 for tok in tokens if tok.length > 0)
    return grafted / len(tokens)


def ratio_sum_lengths(tokens: Sequence[Token]) -> float:
    sum_e = sum(tok.length for tok in tokens if tok.chem == "E" and tok.length > 0)
    sum_s = sum(tok.length for tok in tokens if tok.chem == "S" and tok.length > 0)
    if sum_s > 0:
        return sum_e / sum_s
    return float("inf") if sum_e > 0 else float("nan")


def has_peo(tokens: Sequence[Token]) -> int:
    return int(any(tok.chem == "E" and tok.length > 0 for tok in tokens))


def flip_chemistry_greedy(tokens: Sequence[Token], target_fpeo: float) -> Tuple[List[Token], int]:
    if not 0.0 <= target_fpeo <= 1.0:
        raise ValueError("target_fpeo must be within [0, 1].")

    new_tokens = list(tokens)
    total_mass = sum(tok.length for tok in new_tokens if tok.length > 0)
    if total_mass == 0:
        return new_tokens, 0

    target_sum_e = target_fpeo * total_mass
    current_sum_e = sum(tok.length for tok in new_tokens if tok.chem == "E" and tok.length > 0)

    flips = 0
    while True:
        current_err = abs(current_sum_e - target_sum_e)
        need_increase = current_sum_e < target_sum_e

        best_idx: int | None = None
        best_new_sum_e: float | None = None
        best_err: float | None = None

        for idx, tok in enumerate(new_tokens):
            if tok.length <= 0:
                continue
            if need_increase and tok.chem != "S":
                continue
            if (not need_increase) and tok.chem != "E":
                continue

            delta = tok.length if need_increase else -tok.length
            cand_sum_e = current_sum_e + delta
            cand_err = abs(cand_sum_e - target_sum_e)
            if best_err is None or cand_err < best_err - 1e-12 or (
                abs(cand_err - best_err) <= 1e-12 and tok.length > new_tokens[best_idx].length  # type: ignore[index]
            ):
                best_idx = idx
                best_new_sum_e = cand_sum_e
                best_err = cand_err

        if best_idx is None or best_err is None or best_new_sum_e is None:
            break
        if best_err >= current_err - 1e-12:
            break

        tok = new_tokens[best_idx]
        new_chem = "E" if tok.chem == "S" else "S"
        new_tokens[best_idx] = Token(position=tok.position, chem=new_chem, length=tok.length)
        current_sum_e = best_new_sum_e
        flips += 1

    return new_tokens, flips


def _nonzero_positions(tokens: Sequence[Token]) -> List[int]:
    return [idx for idx, tok in enumerate(tokens) if tok.length > 0]


def permute_nonzero_labels(
    tokens: Sequence[Token],
    labels: Sequence[str],
) -> List[Token]:
    nonzero_idxs = _nonzero_positions(tokens)
    if len(labels) != len(nonzero_idxs):
        raise ValueError("labels length must match number of nonzero tokens.")

    out = list(tokens)
    for out_idx, label in zip(nonzero_idxs, labels, strict=True):
        match = _LABEL_RE.match(label.strip())
        if not match:
            raise ValueError(f"Invalid label: {label!r}")
        chem, length_str = match.groups()
        tok = out[out_idx]
        out[out_idx] = Token(position=tok.position, chem=chem, length=int(length_str))
    return out


def sequence_shuffle(tokens: Sequence[Token], rng: random.Random) -> List[Token]:
    nonzero_idxs = _nonzero_positions(tokens)
    labels = [tokens[idx].label for idx in nonzero_idxs]
    rng.shuffle(labels)
    return permute_nonzero_labels(tokens, labels)


def sequence_blocky(
    tokens: Sequence[Token],
    *,
    start_chem: str = "S",
    rng: random.Random | None = None,
) -> List[Token]:
    nonzero_idxs = _nonzero_positions(tokens)
    labels = [tokens[idx].label for idx in nonzero_idxs]
    s_labels = [lab for lab in labels if lab.startswith("S")]
    e_labels = [lab for lab in labels if lab.startswith("E")]
    if rng is not None:
        rng.shuffle(s_labels)
        rng.shuffle(e_labels)
    if start_chem == "E":
        ordered = e_labels + s_labels
    else:
        ordered = s_labels + e_labels
    return permute_nonzero_labels(tokens, ordered)


def sequence_alternating(tokens: Sequence[Token], *, rng: random.Random | None = None) -> List[Token]:
    nonzero_idxs = _nonzero_positions(tokens)
    labels = [tokens[idx].label for idx in nonzero_idxs]
    s_labels = [lab for lab in labels if lab.startswith("S")]
    e_labels = [lab for lab in labels if lab.startswith("E")]
    if rng is not None:
        rng.shuffle(s_labels)
        rng.shuffle(e_labels)

    if len(s_labels) > len(e_labels):
        next_chem = "S"
    elif len(e_labels) > len(s_labels):
        next_chem = "E"
    else:
        next_chem = rng.choice(["S", "E"]) if rng is not None else "S"

    ordered: List[str] = []
    last_chem: str | None = None
    while s_labels or e_labels:
        preferred = next_chem if last_chem != next_chem else ("E" if next_chem == "S" else "S")

        chosen: str | None = None
        if preferred == "S" and s_labels:
            chosen = s_labels.pop(0)
        elif preferred == "E" and e_labels:
            chosen = e_labels.pop(0)
        elif s_labels:
            chosen = s_labels.pop(0)
        elif e_labels:
            chosen = e_labels.pop(0)

        if chosen is None:
            break
        ordered.append(chosen)
        last_chem = chosen[0]
        next_chem = "E" if last_chem == "S" else "S"

    return permute_nonzero_labels(tokens, ordered)


def remove_grafts(
    tokens: Sequence[Token],
    n_remove: int,
    *,
    pattern: str,
    rng: random.Random | None = None,
) -> Tuple[List[Token], List[int]]:
    if n_remove <= 0:
        return list(tokens), []

    grafted_idxs = _nonzero_positions(tokens)
    if n_remove > len(grafted_idxs):
        raise ValueError("n_remove exceeds grafted site count.")

    selected: List[int] = []
    if pattern == "random":
        if rng is None:
            raise ValueError("rng is required for pattern='random'.")
        selected = rng.sample(grafted_idxs, k=n_remove)
    elif pattern == "periodic":
        backbone_len = len(tokens)
        targets = [(idx + 0.5) * (backbone_len + 1) / n_remove for idx in range(n_remove)]
        available = set(grafted_idxs)
        for target in targets:
            best: int | None = None
            best_dist: float | None = None
            for cand in available:
                dist = abs(tokens[cand].position - target)
                if best_dist is None or dist < best_dist - 1e-12 or (
                    abs(dist - best_dist) <= 1e-12 and tokens[cand].position < tokens[best].position  # type: ignore[index]
                ):
                    best = cand
                    best_dist = dist
            if best is None:
                break
            selected.append(best)
            available.remove(best)
    else:
        raise ValueError("pattern must be 'random' or 'periodic'.")

    out = list(tokens)
    for idx in selected:
        tok = out[idx]
        out[idx] = Token(position=tok.position, chem=tok.chem, length=0)
    return out, sorted(selected)


def scale_peo_lengths(tokens: Sequence[Token], factor: float) -> Tuple[List[Token], float]:
    if factor < 0:
        raise ValueError("factor must be non-negative.")

    out: List[Token] = []
    clipped = 0
    scaled = 0
    for tok in tokens:
        if tok.chem != "E" or tok.length <= 0:
            out.append(tok)
            continue
        scaled += 1
        raw = factor * tok.length
        rounded = int(math.floor(raw + 0.5))
        new_length = min(max(rounded, 1), 10)
        if new_length != rounded:
            clipped += 1
        out.append(Token(position=tok.position, chem=tok.chem, length=new_length))

    pct_clipped = (clipped / scaled) if scaled else 0.0
    return out, pct_clipped


def scale_ps_lengths(tokens: Sequence[Token], factor: float) -> Tuple[List[Token], float]:
    if factor < 0:
        raise ValueError("factor must be non-negative.")

    out: List[Token] = []
    clipped = 0
    scaled = 0
    for tok in tokens:
        if tok.chem != "S" or tok.length <= 0:
            out.append(tok)
            continue
        scaled += 1
        raw = factor * tok.length
        rounded = int(math.floor(raw + 0.5))
        new_length = min(max(rounded, 1), 10)
        if new_length != rounded:
            clipped += 1
        out.append(Token(position=tok.position, chem=tok.chem, length=new_length))

    pct_clipped = (clipped / scaled) if scaled else 0.0
    return out, pct_clipped


def _even_distribution(n: int, total: int, *, min_val: int = 1, max_val: int = 10) -> List[int]:
    if n <= 0:
        return []
    if total < n * min_val or total > n * max_val:
        raise ValueError("Total is infeasible for the given bounds.")

    values = [min_val] * n
    remaining = total - n * min_val
    base_add = remaining // n
    extra = remaining % n
    for idx in range(n):
        values[idx] += base_add + (1 if idx < extra else 0)
    return values


def _max_dispersion_distribution(n: int, total: int, *, min_val: int = 1, max_val: int = 10) -> List[int]:
    if n <= 0:
        return []
    if total < n * min_val or total > n * max_val:
        raise ValueError("Total is infeasible for the given bounds.")

    values = [min_val] * n
    remaining = total - n * min_val
    for idx in range(n):
        if remaining <= 0:
            break
        add = min(max_val - min_val, remaining)
        values[idx] += add
        remaining -= add
    return values


def _mix_distributions(
    uniform: Sequence[int],
    extreme: Sequence[int],
    *,
    alpha: float,
    min_val: int = 1,
    max_val: int = 10,
) -> List[int]:
    if len(uniform) != len(extreme):
        raise ValueError("uniform/extreme lengths mismatch.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1].")

    total = sum(int(v) for v in uniform)
    mixed = [
        int(math.floor(((1.0 - alpha) * u + alpha * e) + 0.5))
        for u, e in zip(uniform, extreme, strict=True)
    ]
    mixed = [min(max(v, min_val), max_val) for v in mixed]

    diff = total - sum(mixed)
    if diff == 0:
        return mixed

    if diff > 0:
        for idx in range(len(mixed)):
            if diff == 0:
                break
            slack = max_val - mixed[idx]
            if slack <= 0:
                continue
            add = min(slack, diff)
            mixed[idx] += add
            diff -= add
    else:
        diff = -diff
        for idx in reversed(range(len(mixed))):
            if diff == 0:
                break
            slack = mixed[idx] - min_val
            if slack <= 0:
                continue
            take = min(slack, diff)
            mixed[idx] -= take
            diff -= take

    if sum(mixed) != total:
        raise RuntimeError("Failed to adjust mixed distribution to the required total.")
    return mixed


def _assign_lengths_min_l1(
    positions_and_old: Sequence[Tuple[int, int]],
    new_lengths: Sequence[int],
) -> dict[int, int]:
    if len(positions_and_old) != len(new_lengths):
        raise ValueError("positions_and_old/new_lengths mismatch.")

    order = sorted(positions_and_old, key=lambda item: (item[1], item[0]))
    new_sorted = sorted(int(v) for v in new_lengths)
    return {pos: new_len for (pos, _), new_len in zip(order, new_sorted, strict=True)}


def redistribute_lengths_within_chemistry(tokens: Sequence[Token], *, alpha: float) -> List[Token]:
    out = list(tokens)
    for chem in ("E", "S"):
        idxs = [idx for idx, tok in enumerate(out) if tok.chem == chem and tok.length > 0]
        if len(idxs) < 2:
            continue

        old_lengths = [out[idx].length for idx in idxs]
        total = sum(old_lengths)
        uniform = _even_distribution(len(old_lengths), total, min_val=1, max_val=10)
        extreme = _max_dispersion_distribution(len(old_lengths), total, min_val=1, max_val=10)
        new_lengths = _mix_distributions(uniform, extreme, alpha=alpha, min_val=1, max_val=10)

        positions_and_old = [(idx, out[idx].length) for idx in idxs]
        assignment = _assign_lengths_min_l1(positions_and_old, new_lengths)
        for idx in idxs:
            tok = out[idx]
            out[idx] = Token(position=tok.position, chem=tok.chem, length=assignment[idx])
    return out
