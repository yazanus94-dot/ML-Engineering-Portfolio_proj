"""24 small Python engineering tasks with golden solutions and pytest tests.

Each task is self-contained: golden source is written to ``solution.py`` and the
test source to ``test_solution.py`` in an isolated temporary directory by the
runner. Task metadata records which evaluation dimension it exercises.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    category: str
    dimensions: tuple[str, ...]
    prompt: str
    golden: str
    tests: str


def _t(task_id, title, category, dimensions, prompt, golden, tests):
    return Task(task_id, title, category, tuple(dimensions), prompt, golden.strip(), tests.strip())


TASKS: tuple[Task, ...] = (
    _t("T01", "Safe division", "robustness", ["correctness", "edge_cases"],
       "Return a/b, or default when b is zero.",
       '''
def safe_divide(a, b, default=0.0):
    if b == 0:
        return default
    return a / b
''',
       '''
from solution import safe_divide
def test_division():
    assert safe_divide(6, 3) == 2.0
def test_zero_denominator():
    assert safe_divide(5, 0, -1.0) == -1.0
'''),
    _t("T02", "Clamp", "numerical", ["correctness", "numerical_stability"],
       "Clamp x into [lo, hi].",
       '''
def clamp(x, lo, hi):
    if lo > hi:
        raise ValueError("lo must be <= hi")
    return min(max(x, lo), hi)
''',
       '''
from solution import clamp
def test_inside():
    assert clamp(5, 0, 10) == 5
def test_outside():
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10
def test_invalid_range():
    try:
        clamp(1, 2, 1)
    except ValueError:
        return
    assert False
'''),
    _t("T03", "Moving average", "numerical", ["correctness", "regression_risk"],
       "Return the trailing moving average of a sequence.",
       '''
def moving_average(values, window):
    if window < 1:
        raise ValueError("window must be positive")
    out = []
    running = 0.0
    for i, value in enumerate(values):
        running += value
        if i >= window:
            running -= values[i - window]
        if i >= window - 1:
            out.append(running / window)
    return out
''',
       '''
from solution import moving_average
def test_basic():
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]
def test_window_one():
    assert moving_average([1, 2], 1) == [1.0, 2.0]
'''),
    _t("T04", "Primality", "algorithms", ["correctness", "edge_cases"],
       "Test whether an integer is prime.",
       '''
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True
''',
       '''
from solution import is_prime
def test_primes():
    assert is_prime(2) and is_prime(97)
def test_non_primes():
    assert not is_prime(1) and not is_prime(9) and not is_prime(-7)
'''),
    _t("T05", "Fibonacci", "algorithms", ["correctness", "performance"],
       "Return the nth Fibonacci number iteratively.",
       '''
def fibonacci(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
''',
       '''
from solution import fibonacci
def test_values():
    assert [fibonacci(n) for n in range(7)] == [0, 1, 1, 2, 3, 5, 8]
def test_large():
    assert fibonacci(50) == 12586269025
'''),
    _t("T06", "Reverse words", "strings", ["correctness", "edge_cases"],
       "Reverse the order of whitespace-separated words.",
       '''
def reverse_words(text):
    return " ".join(text.split()[::-1])
''',
       '''
from solution import reverse_words
def test_basic():
    assert reverse_words("a b  c") == "c b a"
def test_empty():
    assert reverse_words("   ") == ""
'''),
    _t("T07", "Count vowels", "strings", ["correctness"],
       "Count vowels case-insensitively.",
       '''
def count_vowels(text):
    return sum(1 for ch in text.lower() if ch in "aeiou")
''',
       '''
from solution import count_vowels
def test_mixed_case():
    assert count_vowels("Education") == 5
def test_none():
    assert count_vowels("rhythm") == 0
'''),
    _t("T08", "Merge intervals", "algorithms", ["correctness", "edge_cases"],
       "Merge overlapping inclusive intervals.",
       '''
def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result
''',
       '''
from solution import merge_intervals
def test_overlap():
    assert merge_intervals([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]
def test_empty():
    assert merge_intervals([]) == []
'''),
    _t("T09", "Two sum", "algorithms", ["correctness", "performance"],
       "Return indices of two values summing to target.",
       '''
def two_sum(nums, target):
    seen = {}
    for i, value in enumerate(nums):
        need = target - value
        if need in seen:
            return [seen[need], i]
        seen[value] = i
    return []
''',
       '''
from solution import two_sum
def test_found():
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
def test_missing():
    assert two_sum([1, 2], 10) == []
'''),
    _t("T10", "LCS length", "algorithms", ["correctness", "regression_risk"],
       "Longest common subsequence length of two strings.",
       '''
def lcs_length(a, b):
    dp = [0] * (len(b) + 1)
    for ca in a:
        previous = 0
        for j, cb in enumerate(b, 1):
            saved = dp[j]
            dp[j] = previous + 1 if ca == cb else max(dp[j], dp[j - 1])
            previous = saved
    return dp[-1]
''',
       '''
from solution import lcs_length
def test_basic():
    assert lcs_length("ABCBDAB", "BDCABA") == 4
def test_empty():
    assert lcs_length("", "abc") == 0
'''),
    _t("T11", "Matrix transpose", "numerical", ["correctness", "edge_cases"],
       "Transpose a rectangular matrix.",
       '''
def transpose(matrix):
    if not matrix:
        return []
    return [list(row) for row in zip(*matrix)]
''',
       '''
from solution import transpose
def test_rect():
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
def test_empty():
    assert transpose([]) == []
'''),
    _t("T12", "Normalize vector", "numerical", ["correctness", "numerical_stability"],
       "Return the L2-normalized vector; zeros stay zeros.",
       '''
import math
def normalize_vector(values):
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return list(values)
    return [v / norm for v in values]
''',
       '''
from solution import normalize_vector
def test_unit():
    v = normalize_vector([3.0, 4.0])
    assert abs(sum(x*x for x in v) - 1.0) < 1e-12
def test_zero():
    assert normalize_vector([0.0, 0.0]) == [0.0, 0.0]
'''),
    _t("T13", "Stable softmax", "numerical", ["numerical_stability", "correctness"],
       "Numerically stable softmax.",
       '''
import math
def softmax_stable(logits):
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]
''',
       '''
from solution import softmax_stable
def test_large_logits():
    p = softmax_stable([1000.0, 1000.0])
    assert all(math.isfinite(x) for x in p)
    assert abs(sum(p) - 1.0) < 1e-12
def test_shift_invariance():
    a = softmax_stable([1.0, 2.0, 3.0])
    b = softmax_stable([101.0, 102.0, 103.0])
    assert all(abs(x - y) < 1e-9 for x, y in zip(a, b))
'''.replace("from solution import softmax_stable", "import math\nfrom solution import softmax_stable")),
    _t("T14", "RMS", "numerical", ["numerical_stability", "correctness"],
       "Root mean square of a non-empty sequence.",
       '''
import math
def rms(values):
    if not values:
        raise ValueError("empty input")
    return math.sqrt(sum(v * v for v in values) / len(values))
''',
       '''
from solution import rms
def test_basic():
    assert abs(rms([3.0, 4.0]) - 3.5355339059) < 1e-9
def test_empty():
    try:
        rms([])
    except ValueError:
        return
    assert False
'''),
    _t("T15", "Running mean and variance", "numerical", ["numerical_stability", "regression_risk"],
       "Streaming Welford mean/variance of a sequence.",
       '''
def running_mean_var(values):
    count = 0
    mean = 0.0
    m2 = 0.0
    for value in values:
        count += 1
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)
    if count < 2:
        return mean, 0.0
    return mean, m2 / (count - 1)
''',
       '''
from solution import running_mean_var
def test_known():
    mean, var = running_mean_var([1.0, 2.0, 3.0, 4.0])
    assert abs(mean - 2.5) < 1e-12
    assert abs(var - 1.6666666667) < 1e-9
def test_single():
    assert running_mean_var([5.0]) == (5.0, 0.0)
'''),
    _t("T16", "Bag of words", "ml-features", ["correctness", "test_coverage"],
       "Count token frequencies deterministically.",
       '''
from collections import Counter
def bag_of_words(tokens):
    return dict(sorted(Counter(tokens).items()))
''',
       '''
from solution import bag_of_words
def test_counts():
    assert bag_of_words(["a", "b", "a"]) == {"a": 2, "b": 1}
def test_empty():
    assert bag_of_words([]) == {}
'''),
    _t("T17", "Balanced brackets", "strings", ["correctness", "edge_cases"],
       "Check that (), [] and {} are balanced and nested correctly.",
       '''
def brackets_balanced(text):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
''',
       '''
from solution import brackets_balanced
def test_valid():
    assert brackets_balanced("([{}])")
def test_invalid():
    assert not brackets_balanced("([)]")
    assert not brackets_balanced("(")
'''),
    _t("T18", "Rotate right", "algorithms", ["correctness", "regression_risk"],
       "Rotate a list right by k positions.",
       '''
def rotate_right(values, k):
    if not values:
        return []
    k %= len(values)
    return values[-k:] + values[:-k] if k else list(values)
''',
       '''
from solution import rotate_right
def test_basic():
    assert rotate_right([1, 2, 3, 4], 1) == [4, 1, 2, 3]
def test_wrap():
    assert rotate_right([1, 2, 3], 3) == [1, 2, 3]
'''),
    _t("T19", "GCD", "numerical", ["correctness", "performance"],
       "Greatest common divisor of two non-negative integers.",
       '''
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
''',
       '''
from solution import gcd
def test_basic():
    assert gcd(48, 18) == 6
def test_zero():
    assert gcd(0, 7) == 7
'''),
    _t("T20", "Unique sorted", "data", ["correctness", "edge_cases"],
       "Return sorted unique values without mutating the input.",
       '''
def unique_sorted(values):
    return sorted(set(values))
''',
       '''
from solution import unique_sorted
def test_basic():
    source = [3, 1, 2, 1]
    assert unique_sorted(source) == [1, 2, 3]
    assert source == [3, 1, 2, 1]
'''),
    _t("T21", "Binary search", "algorithms", ["correctness", "edge_cases"],
       "Return the index of a target in a sorted list, or -1.",
       '''
def binary_search(sorted_values, target):
    low, high = 0, len(sorted_values) - 1
    while low <= high:
        mid = (low + high) // 2
        value = sorted_values[mid]
        if value == target:
            return mid
        if value < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
''',
       '''
from solution import binary_search
def test_found():
    assert binary_search([1, 3, 5, 7], 5) == 2
def test_missing():
    assert binary_search([1, 3], 2) == -1
'''),
    _t("T22", "Pad sequences", "data", ["correctness", "test_coverage"],
       "Pad or truncate sequences to a fixed length.",
       '''
def pad_sequences(sequences, length, fill=0):
    return [list(seq)[:length] + [fill] * max(0, length - len(seq)) for seq in sequences]
''',
       '''
from solution import pad_sequences
def test_pad_truncate():
    assert pad_sequences([[1, 2], [3, 4, 5]], 3) == [[1, 2, 0], [3, 4, 5]]
'''),
    _t("T23", "Top-k", "data", ["correctness", "performance"],
       "Return the k largest values in descending order.",
       '''
def top_k(values, k):
    if k < 0:
        raise ValueError("k must be non-negative")
    return sorted(values, reverse=True)[:k]
''',
       '''
from solution import top_k
def test_basic():
    assert top_k([5, 1, 9, 3], 2) == [9, 5]
def test_zero():
    assert top_k([1], 0) == []
'''),
    _t("T24", "Group by key", "data", ["correctness", "regression_risk"],
       "Group (key, value) pairs into a dict of lists preserving input order.",
       '''
def group_by_key(pairs):
    grouped = {}
    for key, value in pairs:
        grouped.setdefault(key, []).append(value)
    return grouped
''',
       '''
from solution import group_by_key
def test_basic():
    assert group_by_key([("a", 1), ("b", 2), ("a", 3)]) == {"a": [1, 3], "b": [2]}
def test_empty():
    assert group_by_key([]) == {}
'''),
)


def task_by_id(task_id: str) -> Task:
    return next(task for task in TASKS if task.id == task_id)
