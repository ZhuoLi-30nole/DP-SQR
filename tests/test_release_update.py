"""Publication-label and public-index regressions for the synchronized release."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from dpsqr.display_names import METHOD_LABELS, publication_frame
from dpsqr.methods import sqr_sr
from dpsqr.pipeline import fit_pipeline
from dpsqr.policy import parameters, total_delta
from dpsqr.selection import validate_forced_indices

class ReleaseUpdateTests(unittest.TestCase):
    def test_valid_indices_preserve_values_and_order(self):
        for values in [[], [2, 0], np.array([2, 0]), [2.0, 0.0]]:
            np.testing.assert_array_equal(validate_forced_indices(values, 4), values)

    def test_invalid_indices_are_rejected_before_conversion(self):
        for values in [[1.9], [True], [True, 2], [np.bool_(False), 2], [[1]],
                       [1, 1], [-1], [4], [np.nan], [np.inf], ['1'], [None], [10**100]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_forced_indices(values, 4)

    def test_both_sr_entry_points_reject_fractional_and_boolean_indices(self):
        X = np.column_stack([np.ones(30), np.arange(90.).reshape(30, 3)])
        y = np.arange(30.)
        for values in [[1.9], [True, 2]]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    sqr_sr(X, y, h=.5, gamma=1., eta=.1, steps=1, size=3,
                           forced=values, beta0=np.zeros(4), private=False)
                options = parameters('sqr_sr', n=30, p=4, regime='crime', s=3, intercept=True)
                options['forced'] = values
                with self.assertRaises(ValueError):
                    fit_pipeline('sqr_sr', X, y, options, epsilon=.9,
                                 delta=total_delta(30), tau=.5, seed=1)

    def test_one_display_mapping_for_plots(self):
        from dpsqr.reporting import LABELS
        self.assertIs(LABELS, METHOD_LABELS)
        self.assertEqual(LABELS['tran_dpsgd'], 'Adapted DP-SGD')
        self.assertEqual(LABELS['tran_opm'], 'Adapted OPM')

    def test_publication_frame_does_not_mutate_machine_data(self):
        original = pd.DataFrame({'tran_dpsgd': [1.25], 'tran_opm': [2.5],
                                 'method': ['tran_opm']}, index=['tran_dpsgd'])
        before = original.copy(deep=True)
        displayed = publication_frame(original)
        pd.testing.assert_frame_equal(original, before)
        self.assertEqual(list(displayed.columns), ['Adapted DP-SGD', 'Adapted OPM', 'method'])
        self.assertEqual(list(displayed.index), ['Adapted DP-SGD'])
        self.assertEqual(displayed.loc['Adapted DP-SGD', 'method'], 'Adapted OPM')
        self.assertEqual(displayed.loc['Adapted DP-SGD', 'Adapted DP-SGD'], 1.25)

    def test_pairwise_comparisons_are_labeled_for_publication(self):
        frame = pd.DataFrame({'first': ['sqr_gd'], 'reference': ['tran_opm']})
        displayed = publication_frame(frame)
        self.assertEqual(displayed.iloc[0].tolist(), ['DP-SQR-GD', 'Adapted OPM'])
        self.assertEqual(frame.iloc[0].tolist(), ['sqr_gd', 'tran_opm'])

    def test_table_export_labels_tex_and_preserves_csv_keys(self):
        from make_tables import publish
        frame = pd.DataFrame({'tran_dpsgd': ['.1'], 'tran_opm': ['.2']})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            publish(frame, output, 'example')
            text = (output / 'example.tex').read_text()
            self.assertIn('Adapted DP-SGD', text)
            self.assertIn('Adapted OPM', text)
            self.assertNotIn(r'tran\_', text)
            self.assertEqual(list(pd.read_csv(output / 'example.csv').columns),
                             ['tran_dpsgd', 'tran_opm'])

    def test_tex_labels_are_idempotent_and_preserve_values(self):
        from dpsqr.display_names import relabel_tex
        original = (r"Tran-SGD & tran\_dpsgd & DP-SGD & Adapted DP-SGD & "
                    r"Tran-OPM & tran\_opm & OPM & Adapted OPM & 1.25")
        expected = " & ".join(["Adapted DP-SGD"] * 4 + ["Adapted OPM"] * 4 + ["1.25"])
        self.assertEqual(relabel_tex(original), expected)
        self.assertEqual(relabel_tex(expected), expected)

if __name__ == '__main__':
    unittest.main()
