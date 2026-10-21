#!/usr/bin/env python3
"""Upload ONLY October 20 weight-composition cohort: 15 original runs + 5 mean3."""
import ozstar_submit_weight_composition_15jobs as cohort
import ozstar_upload_5m6m_onefill_results as runner

SUBDIR = 'weight_composition_20261020_3seeds'
TITLE = 'Mixer weight composition (October 20 cohort only)'


def main(argv=None):
    previous = runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE
    try:
        runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE = cohort, SUBDIR, TITLE
        return runner.main(argv)
    finally:
        runner.cohort, runner.SUBDIR, runner.RESULTS_TITLE = previous


if __name__ == '__main__':
    main()
