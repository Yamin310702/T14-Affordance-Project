# T14 teammate package: read this first

This is a compact review package for the T14 3D affordance prediction research proposal. It includes the full experiment story and the important supporting records, while omitting large raw prediction arrays. The original full archive is preserved separately as `research_archive/T14_experiments_2026-10-11.zip`.

## The project in one sentence

We tested a public OpenAD point-cloud affordance model, found that its predictions depend strongly on text-query wording and can miss a partial-view `pour` region, then tested whether fixed prompt changes transfer across affordances and devices.

## Read in this order

1. `00_READ_THIS_FIRST.md` — this handoff and file map.
2. `01_T14_EXPERIMENT_SUMMARY.md` — complete E01–E18 story, key numbers, caveats, and proposed hypothesis.
3. `02_PROPOSAL_SLIDE_OUTLINE.md` — 10-slide plan for the 8-minute Week 7 proposal.
4. `proposal_guideline.pdf` — professor's assignment instructions. These are requirements, not experimental results.
5. `experiments/` — individual protocols, result reports, run metadata, useful CSV tables, scripts, and selected figures.

## Main takeaways

- OpenAD training-name vs held-out-query 18-class mIoU: partial **40.39% vs 10.77%**, full **35.18% vs 7.87%**.
- The same-input CPU and MPS reruns closely agree, so the query-sensitivity result is not an MPS-only effect.
- A selected `unlock` paraphrase transferred to OpenAD with 12.36% IoU vs 0.09% for the original sentence, but the negative-object any-alarm rate increased from 0.34% to 1.61%.
- Prompt benefits were not uniform across affordances. Espresso/Affogato-weight exploratory runs showed mixed effects and sometimes higher false alarms.
- Affostruction was not run. Do not attribute a result to it.

## What is included and omitted

Included: all archived experiment summaries and protocols, run JSON metadata, CSV metric tables, selected plots/failure images, relevant analysis scripts, proposal materials, and the professor's guideline.

Omitted to keep this handoff compact: large `.npz`/`.npy` raw prediction arrays, full source repositories, model checkpoints, and full datasets. These are not required to understand the findings. Use the separate full archive ZIP if raw arrays are needed. Re-running requires separately obtained datasets, checkpoints, and source repositories; see each `run.json` for expected paths and provenance.

The 2026-10-11 proposal guideline gives a submission deadline of October 15, 2026 at 10:59 AM KST. Confirm team number and member names before final slide export.
