# cado feedback: bug reports and wishes from using tplane in ttt-big-fast

Helen asked on 2026-09-25 that new code try `tplane` (github.com/ilovehhhyn/cado) and that
every point of friction, bug or missing feature come back to her. Each entry says what was
run, what happened, what was expected, and the version. Sent entries are marked.

Setup: upstream clone at `/scratch/gpfs/ARORA/hh9077/cado-upstream` (commit 130bfc9 on
2026-09-25), installed into the project venv with `uv pip install -e .../tplane` (tplane 0.1.0);
Helen's own working copy `/scratch/gpfs/ARORA/hh9077/cado` is not touched. On the laptop
tplane cannot be installed (see 1).

| # | date | what | expected | version | sent |
|---|---|---|---|---|---|
| 1 | 2026-09-25 | `pip install -e tplane` under Python 3.13.9 (the laptop's anaconda, where this project's 268 tests run) is refused: `requires a different Python: 3.13.9 not in '<3.13,>=3.11'`. The project pins `>=3.12` only. | either support 3.13 or state in the README why it is excluded; the pin blocks local tests of any code that imports tplane | tplane 0.1.0, pip | yes, 2026-09-25 |
| 2 | 2026-09-25 | `python -c 'import tplane'` run from the cado repository root imports the empty `tplane/` directory as a namespace package (no attributes, no error) because the real package is `tplane/src/tplane`. Easy to mistake for a working install. | a one-line README note, or a `tplane/__init__.py` that raises with the fix named | tplane 0.1.0 | yes, 2026-09-25 |
| 3 | 2026-09-25 | The Della venv was created by uv and has no pip; `python -m pip install -e` fails with `No module named pip`. Not a cado bug, but the README's install line assumes pip. | mention `uv pip install --python <venv>/bin/python -e tplane` beside the pip line | tplane 0.1.0 | yes, 2026-09-25 |

Wishes (not yet sent): a unit kind for an evaluation sequence and for a probe pair; a way to
record a Slurm job's own exit (TIMEOUT, OOM at the cgroup level) when the unit is the whole
job, since our long runs are one process per job.
