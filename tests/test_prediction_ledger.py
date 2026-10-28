"""Stratified audit of the preregistered sign predictions (no training, no data files)."""
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'experiments/prediction_ledger.json'


def load():
    data = json.loads(LEDGER.read_text(encoding='utf-8'))
    thresholds = data['thresholds']
    rows = []
    for p in data['predictions']:
        thr = thresholds[p['architecture']]
        rows.append(dict(p, threshold=thr,
                         rule_sign='+' if p['fluctuation_energy'] >= thr else '-',
                         measured_sign='+' if p['measured'] > 0 else '-'))
    return data, rows


class LedgerIntegrityTest(unittest.TestCase):
    def test_every_prediction_follows_the_stated_rule(self):
        """The recorded sign must be reproducible from the threshold, not typed by hand."""
        _, rows = load()
        for r in rows:
            self.assertEqual(r['predicted'], r['rule_sign'],
                             f"{r['family']} {r['setting']} {r['architecture']}")

    def test_headline_counts(self):
        _, rows = load()
        self.assertEqual(len(rows), 55)
        self.assertEqual(sum(r['predicted'] == r['measured_sign'] for r in rows), 52)

    def test_thresholds_are_the_never_refitted_ones(self):
        data, _ = load()
        self.assertEqual(data['thresholds'],
                         {'deeponet_mlp': 0.2477, 'deeponet_fourier': 0.2728, 'fno': 0.1502})


class StratificationTest(unittest.TestCase):
    def setUp(self):
        _, rows = load()
        for r in rows:
            r['saturated'] = r['measured'] >= 0.9
            r['informative'] = not r['saturated']
            r['boundary'] = abs(r['fluctuation_energy'] - r['threshold']) / r['threshold'] <= 0.15
            r['resolvable'] = abs(r['measured']) > 2.0 * r['sd']
            r['correct'] = r['predicted'] == r['measured_sign']
        self.rows = rows

    def test_saturated_is_signed_not_absolute(self):
        """A large negative R^2 is a decisive failure and must not count as saturated."""
        self.assertTrue(any(r['measured'] < -0.9 and r['informative'] for r in self.rows))

    def test_stratum_sizes(self):
        self.assertEqual(sum(r['saturated'] for r in self.rows), 25)
        self.assertEqual(sum(r['informative'] for r in self.rows), 30)
        self.assertEqual(sum(r['boundary'] for r in self.rows), 6)
        self.assertEqual(sum(r['resolvable'] for r in self.rows), 47)

    def test_accuracy_within_each_stratum(self):
        sel = lambda pred: [r for r in self.rows if pred(r)]  # noqa: E731
        self.assertEqual(sum(r['correct'] for r in sel(lambda r: r['saturated'])), 25)
        self.assertEqual(sum(r['correct'] for r in sel(lambda r: r['informative'])), 27)
        self.assertEqual(sum(r['correct'] for r in sel(lambda r: r['boundary'])), 5)
        self.assertEqual(sum(r['correct'] for r in sel(lambda r: r['resolvable'])), 46)
        both = sel(lambda r: r['informative'] and r['resolvable'])
        self.assertEqual((len(both), sum(r['correct'] for r in both)), (22, 21))

    def test_always_positive_baseline(self):
        """The rule predicted '+' 44 times, so this is the baseline a skeptic will reach for."""
        self.assertEqual(sum(r['predicted'] == '+' for r in self.rows), 44)
        self.assertEqual(sum(r['measured_sign'] == '+' for r in self.rows), 41)
        informative = [r for r in self.rows if r['informative']]
        self.assertEqual(sum(r['measured_sign'] == '+' for r in informative), 16)

    def test_all_misses_are_on_one_operator(self):
        misses = {r['family'] for r in self.rows if not r['correct']}
        self.assertEqual(misses, {'cfd'})

    def test_two_of_three_misses_are_not_resolvable(self):
        misses = [r for r in self.rows if not r['correct']]
        self.assertEqual(len(misses), 3)
        self.assertEqual(sum(r['resolvable'] for r in misses), 1)


class ManuscriptTableTest(unittest.TestCase):
    """The stratified table in main.md must equal the recomputed strata, not a hand copy."""

    def setUp(self):
        path = ROOT / 'paper/main.md'
        if not path.exists():
            self.skipTest('manuscript not present')
        self.text = path.read_text(encoding='utf-8')
        _, rows = load()
        for r in rows:
            r['saturated'] = r['measured'] >= 0.9
            r['informative'] = not r['saturated']
            r['boundary'] = abs(r['fluctuation_energy'] - r['threshold']) / r['threshold'] <= 0.15
            r['resolvable'] = abs(r['measured']) > 2.0 * r['sd']
            r['correct'] = r['predicted'] == r['measured_sign']
        self.rows = rows

    def _row(self, label, predicate):
        sel = [r for r in self.rows if predicate(r)]
        ok = sum(r['correct'] for r in sel)
        lo, hi = WilsonTest.wilson(ok, len(sel))
        # matched to the formatting used in section 5.16 of main.md
        row = (f"| {label} | {len(sel)} | {ok} | {ok / len(sel) * 100:.1f}% | "
               f"[{lo * 100:.1f}, {hi * 100:.1f}] |")
        self.assertIn(row, self.text, row)

    def test_stratified_rows_are_in_the_manuscript(self):
        self._row('all', lambda r: True)
        self._row('saturated (measured `R^2 >= 0.9`)', lambda r: r['saturated'])
        self._row('informative (not saturated)', lambda r: r['informative'])
        self._row('boundary (within 15% of the threshold)', lambda r: r['boundary'])
        self._row('resolvable (`|R^2| > 2 sd`)', lambda r: r['resolvable'])
        self._row('not resolvable', lambda r: not r['resolvable'])
        both = lambda r: r['informative'] and r['resolvable']  # noqa: E731
        self._row('informative **and** resolvable', both)

    def test_baseline_row_is_in_the_manuscript(self):
        informative = [r for r in self.rows if r['informative']]
        ok = sum(r['measured_sign'] == '+' for r in informative)
        lo, hi = WilsonTest.wilson(ok, len(informative))
        self.assertIn(f"**16 of 30 (53.3%, CI [{lo * 100:.1f}, {hi * 100:.1f}])**", self.text)


class WilsonTest(unittest.TestCase):
    @staticmethod
    def wilson(k, n, z=1.96):
        p = k / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return centre - half, centre + half

    def test_intervals_are_valid_and_contain_the_point_estimate(self):
        for k, n in ((52, 55), (41, 55), (27, 30), (16, 30), (5, 6), (0, 3), (3, 3)):
            lo, hi = self.wilson(k, n)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)
            self.assertLessEqual(lo, k / n + 1e-12)
            self.assertGreaterEqual(hi, k / n - 1e-12)

    def test_degenerate_counts_do_not_produce_empty_intervals(self):
        lo, hi = self.wilson(0, 3)
        self.assertGreater(hi, 0.0)
        lo, hi = self.wilson(3, 3)
        self.assertLess(lo, 1.0)


if __name__ == '__main__':
    unittest.main()
