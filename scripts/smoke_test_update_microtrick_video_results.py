#!/usr/bin/env python3
"""Video schema/path/hash/milestone checks with synthetic bytes; no W&B/network."""
import base64
import hashlib
import tempfile
from pathlib import Path

from ozstar_update_microtrick_video_results import video_payloads


def fails(rows, root, end, fragment):
    try:
        video_payloads(rows, root, end)
    except (RuntimeError, ValueError) as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError('Expected rejection: '+fragment)


def main():
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-micro-video-audit-') as temp:
        root = Path(temp)
        # This tests byte verification, not MP4 decoding/experimental footage.
        content = b'SYNTHETIC_SCHEMA_TEST_NOT_EXPERIMENT_VIDEO'
        path = 'media/videos/episode_01_1000004_fake.mp4'
        target = root/'files'/path
        target.parent.mkdir(parents=True)
        with target.open('wb') as handle:
            handle.write(content)
        prefix = 'test_battle_video/episode_01'
        metadata = dict(_type='video-file', path=path)
        nested = {'_step': 1000004, prefix: metadata}
        flat = {'_step': 1000004, prefix+'._type': 'video-file', prefix+'.path': path}
        digest = hashlib.md5(content)
        expected = {path: (len(content), {digest.hexdigest(), base64.b64encode(digest.digest()).decode()})}
        assert video_payloads([nested], root, 1000500) == expected
        assert video_payloads([flat], root, 1000500) == expected
        assert video_payloads([nested, flat], root, 1000500) == expected
        envstep = {k: v for k, v in flat.items() if k != '_step'}
        envstep['env_steps'] = 1000004
        assert video_payloads([envstep], root, 1000500) == expected
        fails([], root, 1000500, 'found []')
        fails([nested], root, 2000500, 'Expected one video')
        fails([nested, dict(nested, _step=1000040)], root, 1000500, 'reused')
        fails([dict(nested, **{prefix: dict(metadata, path='../escape.mp4')})], root, 1000500, 'Unexpected video path')
        fails([dict(nested, **{prefix: dict(metadata, path='/tmp/escape.mp4')})], root, 1000500, 'Unexpected video path')
        fails([dict(nested, _step=True)], root, 1000500, 'valid environment step')
        full = []
        for milestone in range(1, 11):
            relative = 'media/videos/episode_01_{}_fake.mp4'.format(milestone)
            with (root/'files'/relative).open('wb') as handle:
                handle.write(content+bytes([milestone]))
            full.append({'_step': milestone*1000000+4, prefix+'._type': 'video-file',
                         prefix+'.path': relative})
        assert len(video_payloads(full, root, 10050000)) == 10
        fails(full[:-1], root, 10050000, 'Expected one video')
        other = root/'outside.mp4'
        with other.open('wb') as handle:
            handle.write(content)
        linked = root/'files/media/videos/linked.mp4'
        linked.symlink_to(other)
        fails([dict(nested, **{prefix: dict(metadata, path='media/videos/linked.mp4')})],
              root, 1000500, 'Unexpected video path')
    print('PASS: dict/flattened metadata; hashes; env_steps; dedup; missing/reused clips; path escapes; no uploads')


if __name__ == '__main__':
    main()
