#!/bin/bash
#
# Nightly clone, build, test, CDash submit, and email summary for LCM.
# Canonical copy: LCM/doc/lcm/build/nightly.sh. The nightly hosts (algol,
# proxima) run a copy at $LCM_DIR/nightly.sh; keep them in step, since the
# two drifted apart between 2026-06-24 and 2026-08-31.
# Submits to the "Albany-LCM" project at albany-lcm-cdash.sandia.gov
# (configured in LCM/doc/lcm/build/CTestConfig.cmake) and emails a
# parsed summary to amota@sandia.gov via smtp.sandia.gov.
#

set -u

LCM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$LCM_DIR"

export LCM_DIR
export MODULEPATH=$LCM_DIR/LCM/doc/lcm/modulefiles:$MODULEPATH

# Refuse to run concurrently with another nightly instance: on
# 2026-06-09/10 a manual evening run overlapped the 00:00 cron run,
# racing on the shared build/install directories and submitting a
# failed LCM configure (Trilinos install momentarily absent) into the
# same CDash build (both runs fell in the same 01:00 UTC stamp window).
exec 9>"$LCM_DIR/.nightly.lock"
if ! flock -n 9; then
    echo "another nightly.sh instance holds $LCM_DIR/.nightly.lock; exiting" >&2
    exit 1
fi

HOST=$(hostname -s)
LOG_DIR=$LCM_DIR/nightly-logs
mkdir -p "$LOG_DIR"
RUN_LOG=$LOG_DIR/$(date +%Y%m%d).log

{
  echo "=== LCM Nightly ==="
  echo "Date: $(date -Iseconds)"
  echo "Host: $(hostname)"
  echo "LCM_DIR: $LCM_DIR"
  echo

  echo "--- Cloning repositories ---"
  for REPO_INFO in \
      "Trilinos|git@github.com:trilinos/Trilinos.git|develop" \
      "LCM|git@github.com:sandialabs/LCM.git|main"
  do
      IFS='|' read -r NAME URL BRANCH <<< "$REPO_INFO"
      echo "Cloning $NAME ($BRANCH)..."
      rm -rf "$NAME"
      git clone -q -b "$BRANCH" "$URL" "$NAME" 2>&1 | tail -1
  done
  echo

  ln -sf LCM/doc/lcm/build/lcm .
  NUM_PROCS=$(nproc --all)   # not $(nproc): OMP_NUM_THREADS=1 in login env caps it to 1

  for MODULE in serial-gcc-release serial-clang-release; do
      echo "=== Building with module: $MODULE ==="
      if ./lcm all "$NUM_PROCS" --module="$MODULE" --cdash; then
          RESULT=PASS
      else
          RESULT=FAIL
      fi

      # The driver's exit code is not sufficient on its own. On 2026-08-31
      # the Sacado rename of Kokkos_ViewFactory.hpp made every translation
      # unit fail; no Albany binary was produced, yet snl_build() returned
      # 0, ctest submitted a Build.xml with zero <Error> entries, and the
      # driver reported RESULT=PASS while all 194 runnable tests failed to
      # launch. The root cause is that CTEST_USE_LAUNCHERS is set in the
      # ctest script but never lands in the project cache, so ctest counts
      # no compile errors. Until that is fixed in the driver, verify the
      # one artifact every test actually needs.
      ALBANY_EXE=$LCM_DIR/lcm-build-$MODULE/src/Albany
      if [ -x "$ALBANY_EXE" ]; then
          echo "[$MODULE] BUILD_ARTIFACT=OK ($ALBANY_EXE)"
      else
          echo "[$MODULE] BUILD_ARTIFACT=MISSING ($ALBANY_EXE)"
          RESULT=FAIL
      fi

      echo "[$MODULE] RESULT=$RESULT"
      echo
  done

  echo "=== Nightly complete: $(date -Iseconds) ==="
} 2>&1 | tee "$RUN_LOG"

# Email summary + tail of run log via Sandia internal relay. CDash gets
# the rich result via ctest_submit (driven by --cdash above); this email
# is the lightweight at-a-glance + the place to land if CDash submit
# itself failed.
python3 - "$RUN_LOG" "$HOST" <<'PY'
import re, smtplib, sys
from email.message import EmailMessage
from pathlib import Path

run_log = Path(sys.argv[1])
host    = sys.argv[2]
text    = run_log.read_text()
lines   = text.splitlines()

# Per-module pass/fail. The shell RESULT comes from the lcm ctest -S
# driver's exit code, which is 0 even when tests fail (dashboard mode),
# so cross-check against the ctest test summaries: one
# "N% tests passed, M tests failed out of T" line per module, in order.
results = re.findall(r'\[(\S+)\] RESULT=(\w+)', text)
test_sums = re.findall(r'(\d+)% tests passed, (\d+) tests failed out of (\d+)', text)
merged = []
for i, (m, v) in enumerate(results):
    if i < len(test_sums):
        pct, nfail, ntot = test_sums[i]
        if int(nfail) > 0:
            v = 'FAIL'
        merged.append((m, f'{v} ({int(ntot)-int(nfail)}/{ntot} tests)'))
    else:
        merged.append((m, f'{v} (no test summary found)'))
results_display = merged
all_pass = bool(results) and all(v == 'PASS' for _, v in results) \
    and bool(test_sums) and all(int(nf) == 0 for _, nf, _ in test_sums) \
    and len(test_sums) >= len(results)   # every module must have produced a test summary

# CDash submission health. A ctest_submit that exhausts its retries logs
# "Error when uploading file: <path>" and leaves that build's dashboard
# row missing a stage. A dropped Configure.xml, for instance, renders as a
# failed/incomplete row on CDash even though every test passed -- exactly
# the case where a test-only summary emails PASS while CDash shows red
# (proxima lcm-clang-release, 2026-07-17: transient CDash outage dropped
# the Configure.xml). Surface it as WARN and name the orphaned parts; each
# is re-submittable with:  curl --upload-file <part> '<submit-url-from-log>'
#
# A part that failed one ctest_submit pass may still land on a later one;
# ctest logs "Uploaded: <path>" only on success, so treat a part as truly
# dropped only when no successful upload of that same path appears anywhere
# in the run (avoids over-warning on a transient blip that self-recovered).
# A build that produced no binary is the loudest possible failure and must
# not be reported as a pile of test regressions: name it first, so the
# reader does not go hunting through 194 launch failures for a cause.
missing_artifacts = re.findall(r'\[(\S+)\] BUILD_ARTIFACT=MISSING \((\S+)\)', text)

failed_uploads = re.findall(r'Error when uploading file:\s*(\S+)', text)
uploaded_ok    = set(re.findall(r'Uploaded:\s*(\S+)', text))
submit_failures = [p for p in dict.fromkeys(failed_uploads) if p not in uploaded_ok]

if not all_pass:
    status_word = 'FAIL'
elif submit_failures:
    status_word = 'WARN'
else:
    status_word = 'PASS'
summary_line = '  ' + '\n  '.join(f'{m}: {v}' for m, v in results_display) if results_display else '  (no module ran)'
build_line = ('  every module produced an Albany binary'
              if not missing_artifacts
              else '  NO BINARY -- the build failed. Every test failure below is a\n'
                   '  launch failure, not a regression:\n    ' +
                   '\n    '.join(f'{m}: {p}' for m, p in missing_artifacts))
submit_line = ('  all dashboard parts submitted'
               if not submit_failures
               else '  DROPPED -- CDash row(s) incomplete, re-submit these XML parts:\n    ' +
                    '\n    '.join(submit_failures))

# Per-module CTest summary block + failure lines
ctest_blocks = []
i = 0
while i < len(lines):
    if 'Total Test time' in lines[i]:
        start = max(0, i - 10)
        end   = min(len(lines), i + 1)
        while end < len(lines) and ('(Failed)' in lines[end] or 'FAILED' in lines[end]
                                    or lines[end].strip().startswith('\t')):
            end += 1
        ctest_blocks.append('\n'.join(lines[start:end]))
        i = end
    else:
        i += 1
ctest_summary = '\n\n'.join(ctest_blocks) if ctest_blocks else '(no CTest summary parsed)'

body = (
    f'Host:    {host}\n'
    f'Run log: {run_log}\n'
    f'CDash:   https://albany-lcm-cdash.sandia.gov/index.php?project=Albany-LCM\n'
    f'\n=== Results ===\n{summary_line}\n'
    f'\n=== Build artifacts ===\n{build_line}\n'
    f'\n=== Dashboard submission ===\n{submit_line}\n'
    f'\n=== CTest summary per module ===\n{ctest_summary}\n'
    f'\n=== last 200 lines of run log ===\n' + '\n'.join(lines[-200:])
)
m = EmailMessage()
m['Subject'] = f'[lcm nightly {host}] {status_word}'
m['From']    = f'lcm-nightly@{host}.sandia.gov'
m['To']      = 'amota@sandia.gov'
m.set_content(body)
try:
    with smtplib.SMTP('smtp.sandia.gov', 25, timeout=30) as s:
        s.send_message(m)
    print(f'emailed nightly summary to amota@sandia.gov (status={status_word})')
except Exception as e:
    print(f'WARN: email failed: {e}')
PY
