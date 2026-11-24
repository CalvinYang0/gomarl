"""Validate predeclared user-heatmap masks, never infer importance from colors at runtime."""
import hashlib
import json
from pathlib import Path


def load_heatmap_mask(path, map_name, state_dim):
    raw = Path(path).read_bytes()
    bundle = json.loads(raw)
    if bundle.get('version') != 1 or bundle.get('fill_value') != 1:
        raise ValueError('Unsupported fixed-mask bundle or fill convention')
    record = bundle.get('records', {}).get(map_name)
    if not isinstance(record, dict) or record.get('state_dim') != state_dim:
        raise ValueError('Fixed mask is missing or has the wrong map/state dimension')
    closed = record.get('closed_indices')
    if (not isinstance(closed, list) or not closed or len(set(closed)) != len(closed) or
            any(type(i) is not int or not 0 <= i < state_dim for i in closed) or
            record.get('source_seed') != 3 or record.get('source_step', 0) < 5000000):
        raise ValueError('Invalid frozen heatmap indices/provenance')
    mask = [int(i not in closed) for i in range(state_dim)]
    return mask, dict(record, bundle_sha256=hashlib.sha256(raw).hexdigest(),
                      mask_sha256=hashlib.sha256(bytes(mask)).hexdigest())
