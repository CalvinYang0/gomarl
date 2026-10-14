#!/usr/bin/env python3
"""Offline safety tests; never contact Slurm or delete real experiment data."""
import json
from pathlib import Path
import tempfile
import unittest

import ozstar_clean_except_qmix_variants as c


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'ozstar_logs').mkdir()
        (self.root / 'wandb').mkdir()

    def run_data(self, name, group, job):
        sacred = self.root / 'results' / 'sacred' / '5m_vs_6m' / group / str(job)
        sacred.mkdir(parents=True)
        (sacred / 'config.json').write_text(json.dumps(dict(wandb_run_name=name, name=group)))
        wandb = self.root / 'wandb' / ('offline-run-20261001_120000-r' + str(job))
        (wandb / 'files').mkdir(parents=True)
        (wandb / 'files' / 'config.yaml').write_text(
            'wandb_run_name:\n  value: ' + name + '\nname:\n  value: ' + group + '\n')
        (self.root / 'ozstar_logs' / (name + '_' + str(job) + '.out')).write_text(str(wandb) + '\n')
        artifact = self.root / 'results' / 'models' / (group + '__2026-10-01_12-00-00')
        artifact.mkdir(parents=True, exist_ok=True)
        return sacred, wandb, artifact

    def test_preserves_all_variants_and_retries(self):
        paths = []
        job = 1
        for variant in ('static10', 'group_qchange', 'group_tderror'):
            for seed in (1, 2, 3):
                for retry in ('', '_retry1'):
                    paths.extend(self.run_data('smac_5m6m_qmix_state_{}_10m_s{}_mixercontrol{}'.format(
                        variant, seed, retry), c.KEEP_GROUP, job))
                    job += 1
        targets, kept = c.inventory(self.root, set(), lambda j: True)
        self.assertEqual(targets, [])
        self.assertTrue(set(paths) <= set(kept))

    def test_live_unknown_and_mixed_group(self):
        ended = self.run_data('baseline_s1', 'baseline', 1)
        active = self.run_data('baseline_s2', 'baseline', 2)
        unknown = self.run_data('unknown_s1', 'unknown', 3)
        targets, kept = c.inventory(self.root, {'2'}, lambda j: j != '3')
        self.assertIn(ended[0], targets)
        self.assertIn(ended[1], targets)
        self.assertIn(ended[2], kept)  # Shared group with active seed remains.
        self.assertTrue(set(active + unknown) <= set(kept))

    def test_ended_order_and_incremental_reindex(self):
        sacred, wandb, artifact = self.run_data('ordinary_qmix_s1', 'old', 1)
        targets, _ = c.inventory(self.root, set(), lambda j: True)
        self.assertEqual(targets, [artifact, wandb, sacred])
        for path in targets:
            current, _ = c.inventory(self.root, set(), lambda j: True)
            self.assertIn(path, current)
            import shutil
            shutil.rmtree(path)

    def test_unresolved_live_blocks_shared_artifact(self):
        sacred, wandb, artifact = self.run_data('ordinary_vdn_s1', 'old', 1)
        targets, kept = c.inventory(self.root, {'999'}, lambda j: True)
        self.assertIn(artifact, kept)
        self.assertIn(sacred, targets)

    def test_pending_retry_without_logs_protects_same_name(self):
        paths = self.run_data('baseline_s1', 'old', 1)
        targets, kept = c.inventory(self.root, {'999'}, lambda j: True, {'baseline_s1'})
        self.assertEqual(targets, [])
        self.assertTrue(set(paths) <= set(kept))

    def test_new_requested_cohort_survives_after_completion(self):
        paths = self.run_data('smac_8m9m_qmix_baseline_10m_s1_fresh20261014',
                              'qmix_vdn_calibration_gate_10m_3seeds_fresh20261014', 1)
        targets, kept = c.inventory(self.root, set(), lambda j: True)
        self.assertEqual(targets, [])
        self.assertTrue(set(paths) <= set(kept))

    def test_factorial_ablation_is_preserved_after_completion(self):
        paths = self.run_data('smac_5m6m_qmix_state_group_calibration_shared_10m_s1_ablate20261014',
                              '5m6m_score_gradient_factorial_10m_6jobs_20261014', 1)
        targets, kept = c.inventory(self.root, set(), lambda j: True)
        self.assertEqual(targets, [])
        self.assertTrue(set(paths) <= set(kept))

    def test_symlink_is_not_followed(self):
        sacred, wandb, _ = self.run_data('old_s1', 'old', 1)
        protected = self.root / 'protected.txt'
        protected.write_text('keep me')
        (wandb / 'files' / 'saved.json').symlink_to(protected)
        state, _ = c.snapshot(wandb, self.root)
        self.assertIn('files/saved.json', state)
        import shutil
        shutil.rmtree(wandb)
        self.assertEqual(protected.read_text(), 'keep me')
        with self.assertRaises(RuntimeError):
            c.snapshot(self.root / 'results', self.root)
        link = self.root / 'results' / 'linked'
        link.symlink_to(sacred, target_is_directory=True)
        with self.assertRaises(RuntimeError):
            c.snapshot(link / 'config.json', self.root)

    def test_clean_three_ablation_cohort_is_preserved(self):
        paths = []
        for job, label in enumerate(('qchange_plainmask', 'variance'), 50):
            paths.extend(self.run_data('smac_5m6m_qmix_state_group_{}_10m_s1_ablate20261015'.format(label),
                '5m6m_qchange_three_ablation_10m_9jobs_20261015', job))
        targets, kept = c.inventory(self.root, set(), lambda j: True)
        self.assertEqual(targets, [])
        self.assertTrue(set(paths) <= set(kept))

    def test_8m9m_gate_regularization_cohort_is_preserved(self):
        paths = self.run_data('smac_8m9m_qmix_state_static10_variance_10m_s1_ablate20261015',
                              '8m9m_ones_gate_regularization_10m_9jobs_20261015', 60)
        targets, kept = c.inventory(self.root, set(), lambda j: True)
        self.assertEqual(targets, [])
        self.assertTrue(set(paths) <= set(kept))


if __name__ == '__main__':
    unittest.main()
