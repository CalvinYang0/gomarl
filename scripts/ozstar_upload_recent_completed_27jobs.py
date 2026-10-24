#!/usr/bin/env python3
"""Update exactly October 22 reruns + October 23 refresh, not arbitrary recents.

Forward the ordinary uploader flags to each independent batch. A failed batch
does not prevent attempting the other; any failure produces a nonzero exit.
No training, data deletion, original-failed-attempt upload or history stitching.
"""
import sys

import ozstar_upload_qtot_soft_rerun_results as rerun
import ozstar_upload_mixer_state_refresh_results as refresh


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    errors, reports = [], []
    for module in (rerun, refresh):
        print('\nBATCH: ' + module.cohort.GROUP, flush=True)
        try:
            reports.append(module.main(args))
        except (Exception, SystemExit) as exc:
            if isinstance(exc, SystemExit) and exc.code == 0:
                return reports
            errors.append(module.cohort.GROUP + ': ' + str(exc))
            print('BATCH INCOMPLETE: ' + errors[-1], flush=True)
    if errors:
        raise SystemExit('PARTIAL/INCOMPLETE; review each batch upload_report.json: ' + '; '.join(errors))
    count = sum(len(r['mean3']) for r in reports)
    label = 'LOCAL VALIDATION COMPLETE' if '--no-upload' in args else 'BOTH BATCHES COMPLETE'
    print(label + ': {} mean3 uploaded; {} map/model groups prepared'.format(
        count, sum(len(r['mean3_plan']) for r in reports)), flush=True)
    return reports


if __name__ == '__main__':
    main()
