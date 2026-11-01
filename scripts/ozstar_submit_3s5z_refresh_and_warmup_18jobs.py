#!/usr/bin/env python3
"""One command for the two exact 9-job cohorts; preserves old job identities."""
import argparse
import ozstar_submit_3s5z_soft_refresh_live_9jobs as refresh
import ozstar_submit_3s5z_onefill_warmup_9jobs as warmup


def build_plans(repo):
    plans = refresh.build_plans(repo)+warmup.build_plans(repo)
    if len(plans) != 18 or len({p['job_name'] for p in plans}) != 18:
        raise RuntimeError('Expected six variants x three seeds, distinct identities')
    return plans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('all', 'refresh', 'warmup'), default='all')
    args = parser.parse_args(argv)
    # Each existing cohort keeps its own quota guard, lock, test-only preflight,
    # exact-name retention and receipt. Never rename old jobs into a new cohort.
    if args.suite in {'all', 'refresh'}:
        refresh.main([])
    if args.suite in {'all', 'warmup'}:
        warmup.main([])


if __name__ == '__main__':
    main()
