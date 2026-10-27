# T14 experiment archive

Saved 2026-10-05. Original working files are unchanged. This archive groups completed experiments separately from planned work.

| ID | Experiment | Main evidence |
|---|---|---|
| E01 | Original OpenAD full/partial baselines | Original metric logs; 14.4121% / 12.4964% mIoU |
| E02 | Selected OpenAD failures | Three figures, analysis scripts, original user progress report |
| E03 | CPU OpenAD check | One-sample diagnostics and metadata |
| E04 | MPS OpenAD check | One-sample diagnostics and metadata |
| E05 | Dataset-wide OpenAD analysis | All 9,140 views; 12.5364% mIoU; confusion matrices, per-class and sample tables |
| E06 | Espresso compatibility checks | CPU encoder and MPS encoder/decoder checks with random weights/synthetic text |
| E07 | Pretrained unofficial Espresso inference | Real vase input, new heatmap, figure, run log, Hub revisions and scripts |
| P01 | Three-model comparison | Pilot manifest and plan only; no Affostruction results |

## How to interpret the records

- E01 and E05 are different runs; do not replace one with the other. Batch size and stochastic point sampling differ.
- E02's original score values come from the user progress report; original per-case stdout logs were not located. The preserved analysis script is its current version, configured for pour sample 452, not a historical snapshot of each run.
- E06 is an execution test, not an accuracy result. Its current script tests more than the earlier CPU-only encoder run.
- E07 is an unofficial Espresso implementation on an out-of-protocol sample. The 1.643-second observation is a single inference measurement after loading, not a benchmark. Recorded MPS allocation is current allocation, not peak memory.
- Affostruction has not run. No three-model accuracy comparison is complete.
- Some setup attempts failed before the successful Espresso run; current saved logs are the final successful run. Earlier transient logs were overwritten and have not been reconstructed.

## Reproducibility materials

`experiment_register.csv` lists status/results/limitations. `code/` preserves source snapshots and Git diffs captured at archive time; these are not proof of exact source state at every historical run. Run-specific metadata takes precedence. `copied_file_manifest.json` maps copied evidence to its original absolute path and checksum. `SHA256SUMS.json` covers all archive files except itself. `external_resources.json` records large resources kept in their existing locations, with checkpoint hashes. Large weights, datasets, virtual environments and downloaded dependencies are not duplicated in this archive; the archive is not a standalone runnable distribution.

Open raw figures and CSVs inside the corresponding experiment folder. Original scripts may contain absolute paths to this computer; adapt those paths when reproducing elsewhere.
