#!/usr/bin/env python3
"""Verify official Workspace/MediaBrowser model offline; no W&B writes."""
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import tempfile
import json

from wandb_seed_figure_sections import make_workspace, upload_sectioned


def main():
    records = [dict(section=section, model='model_' + str(i), title='Model ' + str(i),
                    path='synthetic.png', key=section + '/' + str(i))
               for section, count in [('5m6m', 10), ('8m9m', 6)] for i in range(count)]
    summaries = [('5m6m mean', 'Media/5m6m_mean'), ('8m9m mean', 'Media/8m9m_mean')]
    workspace = make_workspace('synthetic', 'project', 'abc1234', records, summaries)
    assert [s.name for s in workspace.sections] == ['Media', '5m6m', '8m9m']
    assert [len(s.panels) for s in workspace.sections] == [2, 10, 6]
    single_map = make_workspace('synthetic', 'project', 'abc1234', records[:4], summaries[:1])
    assert [s.name for s in single_map.sections] == ['Media', '5m6m']
    assert all(s.layout_settings.columns == 1 for s in workspace.sections)
    assert workspace.auto_generate_panels is False
    for section in workspace.sections[1:]:
        assert section.is_open
        assert all(p.media_keys[0].startswith(section.name + '/') for p in section.panels)
    # Validate serialized schema with the official SDK, without save/network.
    wire = workspace._to_model().model_dump(by_alias=True)
    assert wire['spec']['section']['runSets'][0]['filters']
    with tempfile.TemporaryDirectory() as temp:
        logged = {}
        class Run:
            url = 'https://wandb.ai/synthetic/project/runs/abc1234'
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def log(self, payload):
                logged.update(payload)
            def log_artifact(self, artifact):
                pass
        charts = SimpleNamespace(GROUP='synthetic', TARGET_STEPS=10000000,
                                 FIELDS=['seed'], inventory_table_data=lambda rows: [])
        with patch('wandb.init', return_value=Run()), patch('wandb.Image', side_effect=lambda *a, **k: 'image'), \
             patch('wandb.Table', return_value='table'), patch('wandb.Artifact'), \
             patch('wandb.sdk.lib.runid.generate_id', return_value='abc1234'), \
             patch('wandb_seed_figure_sections.make_workspace') as maker:
            maker.return_value.save.return_value = SimpleNamespace(url='https://wandb.ai/synthetic/project?nw=test')
            upload_sectioned('synthetic/project', Path(temp),
                             {'smac_5m6m': ['mean.png'], 'smac_8m9m': ['mean.png']}, [], 10, charts, records)
            assert set(r['key'] for r in records) <= set(logged)
            assert 'smac_5m6m_individual_seeds' not in logged  # No tiny composite upload.
            assert json.loads((Path(temp) / 'model_panel_upload.json').read_text())['workspace_url']
    print('PASS: explicit Media/5m6m/8m9m sibling sections; 10+6 ordered panels; single-map ablation view; exact run filter; upload payload; no network')


if __name__ == '__main__':
    main()
