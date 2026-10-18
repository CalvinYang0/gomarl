#!/usr/bin/env python3
"""Real mixers, archived-score repair, and plot structure; no SC2 or cloud."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from modules.mixers.state_input_control import StateInputControlMixer
from utils.mixer_state_importance import score_state_slots, log_state_snapshot
from utils.mixer_group_visualization import (prepare_group_score_snapshot,
    group_score_panels, render_group_score_snapshot)
import ozstar_backfill_grouping_heatmaps as backfill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', type=Path)
    options = parser.parse_args()
    th.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='gomarl-group-score-QA-') as temp:
        args = SimpleNamespace(mixer_input_control='group_qchange', state_shape=6,
            mixer_input_append_mask=False, n_agents=2, mixing_embed_dim=4,
            env_args={'map_name': 'SYNTHETIC'}, local_results_path=temp,
            unique_token='SYNTHETIC_QA', name='SYNTHETIC_QA',
            wandb_run_name='SYNTHETIC_GROUP_SCORE_QA_NOT_EXPERIMENT', seed=3)
        th.manual_seed(19)
        for append in (False, True):
            args.mixer_input_append_mask = append
            mixer = StateInputControlMixer(args)
            before = th.tensor([True, False, True, False, False, True])
            mixer.important_slots.copy_(before)
            states, q = th.randn(2, 3, 6), th.randn(2, 3, 2)
            targets, valid = th.randn(2, 3, 2), th.ones(2, 3, 1)
            measured, td, count = score_state_slots(mixer, q, states, targets, valid)
            assert count == 6
            # Explicitly probe EVERY field in BOTH actual mixers, including the
            # skipped fields. Inactive fields produce exactly zero changes.
            masks, ref = mixer.masks(states), states.mean((0, 1))
            for group, mask in enumerate(masks):
                base = mixer.conditioned_value(q, states, group, mask)
                for j in range(6):
                    altered = states.clone()
                    altered[..., j] = ref[j]
                    delta = (mixer.conditioned_value(q, altered, group, mask) - base).abs().mean()
                    if not mask[j]:
                        assert delta.item() == 0
                    else:
                        assert th.allclose(delta, measured[j], atol=1e-6)
            mixer.qchange_scores.copy_(measured)
            mixer.tderror_scores.copy_(td)
            after = ~before
            mixer.important_slots.copy_(after)
            migration = dict(important_before=before.tolist(), important_after=after.tolist())
            logger = SimpleNamespace(use_wandb=False, log_stat=lambda *a: None)
            path = log_state_snapshot(mixer, args, logger, 100, 1, migration,
                                      evaluated_partition=before)
            record = json.loads(path.read_text())
            assert record['masks_or_probability'][0] == after.int().tolist()
            assert record['score_masks'][0] == before.int().tolist()
            assert len(group_score_panels(record)) == 4
            for key in ('qchange', 'tderror'):
                groups = record['group_scores'][key]
                assert [a + b for a, b in zip(*groups)] == record[key]
                assert all(v == 0 for row, mask in zip(groups, record['score_masks'])
                           for v, keep in zip(row, mask) if not keep)
            # The same mask snapshot must actually reach the plot (not merely
            # be written to JSON), and each subplot must get the correct bars.
            from matplotlib.axes import Axes
            heatmaps, bars = [], []
            original_imshow, original_bar = Axes.imshow, Axes.bar
            def capture_imshow(axis, x, *a, **kw):
                heatmaps.append(copy.deepcopy(x))
                return original_imshow(axis, x, *a, **kw)
            def capture_bar(axis, x, height, *a, **kw):
                bars.append(list(height))
                return original_bar(axis, x, height, *a, **kw)
            with patch.object(Axes, 'imshow', capture_imshow), patch.object(Axes, 'bar', capture_bar):
                render_group_score_snapshot(record, Path(temp) / 'captured.png')
            assert heatmaps[0] == record['score_masks']
            assert bars[:4] == [panel[3] for panel in group_score_panels(record)]
            # Archived pre-migration masks can be recovered without changing
            # original bytes, model parameters, or the saved scalar scores.
            legacy = copy.deepcopy(record)
            for key in ('score_important_membership', 'score_masks', 'group_scores'):
                legacy.pop(key, None)
            raw = json.dumps(legacy).encode()
            path.write_bytes(raw)
            source = {'id': 'SYNTHETIC_QA', 'name': args.wandb_run_name}
            model_before = {k: v.clone() for k, v in mixer.state_dict().items()}
            root, files, summary = backfill.prepare_score_repair_bundle(
                path.parent, 'mixer_input', source, Path(temp) / 'repair')
            assert path.read_bytes() == raw and len(files) == 2
            fixed = json.loads(files[1].read_text())
            assert fixed['score_masks'][0] == before.int().tolist()
            assert fixed['qchange'] == legacy['qchange']
            assert all(th.equal(v, model_before[k]) for k, v in mixer.state_dict().items())
            assert summary['mixer_input/grouping_heatmap']['path'].startswith('grouping_score_repair_v3/')
            bundle, reused = backfill.reuse_images((root, files, summary), {})
            assert reused == 0 and len(bundle[1]) == 2
            # Exercise the exact two-file upload route with a fake transport,
            # preserving history/scalars and remaining idempotent on retry.
            remote, uploaded = {}, []
            original_summary = {'test_battle_won_mean': .75, '_step': 100}
            run = SimpleNamespace(name=source['name'], state='finished',
                summary=original_summary.copy(), url='https://example.invalid/SYNTHETIC_QA')
            run.files = lambda names, per_page: [SimpleNamespace(name=n,
                md5=remote[n]['md5'], size=remote[n]['size']) for n in names if n in remote]
            api = SimpleNamespace(run=lambda _: run, flush=lambda: None)
            class Transport:
                def __init__(self, project, source_id):
                    assert project == 'SYNTHETIC/QA' and source_id == source['id']
                def urls(self, names):
                    return [], {name: {'name': name} for name in names}
                def transfer(self, file, entry, headers):
                    remote[entry['name']] = backfill.fingerprints(file)
                    uploaded.append(entry['name'])
            ledger, state_path = {}, root / 'qa_ledger.json'
            for _ in range(2):
                assert backfill.upload_bundle(api, 'SYNTHETIC/QA', source, bundle,
                    ledger, state_path, workers=1, uploader_factory=Transport)
            assert len(uploaded) == 2
            assert all(run.summary[k] == v for k, v in original_summary.items())
            unknown = copy.deepcopy(legacy)
            unknown.pop('migration')
            try:
                prepare_group_score_snapshot(unknown)
                raise AssertionError('Unknown score ownership must not be guessed')
            except ValueError:
                pass
            wrong = copy.deepcopy(record)
            wrong['score_important_membership'] = after.tolist()
            try:
                prepare_group_score_snapshot(wrong)
                raise AssertionError('Mismatched scoring partition must fail')
            except ValueError:
                pass
            # All-important/all-unimportant and signed metrics retain zeros
            # outside the owning group; do not clip negative variance scores.
            for variant in ('group_tderror', 'group_calibration', 'group_variance'):
                for selected in ([True] * 6, [False] * 6):
                    example = copy.deepcopy(legacy)
                    example.update(variant=variant, migration=None,
                        masks_or_probability=[selected, [not x for x in selected]],
                        normalized_removal_gain=[-.2] * 6, qtot_variance_difference=[-.3] * 6)
                    fixed = prepare_group_score_snapshot(example)
                    for group, key, label, values in group_score_panels(fixed):
                        assert all(v == 0 for v, keep in zip(values, fixed['score_masks'][group]) if not keep)
                    if variant == 'group_variance':
                        assert min(sum(fixed['group_scores']['qtot_variance_difference'], [])) == -.3
            if options.preview and not append:
                # Production-width layout QA with clearly synthetic values.
                from utils.mixer_state_importance import state_slot_names
                preview = copy.deepcopy(record)
                args.env_args = {'map_name': '5m_vs_6m'}
                args.n_actions = 12
                names = state_slot_names(args, 98)
                chosen = [(i % 7 == 0) for i in range(98)]
                preview.update(state_slot_names=names, migration=None,
                    score_important_membership=chosen,
                    masks_or_probability=[chosen, [not x for x in chosen]],
                    qchange=[(i % 13) * .03 for i in range(98)],
                    tderror=[(i % 11) * .02 for i in range(98)], ranking_ema=[0.] * 98)
                render_group_score_snapshot(preview, options.preview)
                args.env_args = {'map_name': 'SYNTHETIC'}
        print('PASS: both real mixer encodings; masked-field invariance; separate branch bars; '
              'pre/post migration alignment; signed/empty groups; legacy fail-closed; '
              'archived repair; original bytes and model parameters unchanged')


if __name__ == '__main__':
    main()
