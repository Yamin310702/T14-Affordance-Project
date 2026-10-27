# Team 07: T14 3D Affordance Prediction

Course research project on how text-query wording changes 3D affordance localization and false alarms. This repository shares research scripts, experiment records, and the current Team 07 proposal with the team.

## Start here

For setup from a fresh clone, follow the [step-by-step clone and run guide (PDF)](output/pdf/T14_Project_Clone_and_Run_Guide.pdf). It covers the external OpenAD setup, data and checkpoint downloads, path fixes, baseline evaluation, E17/E18, and troubleshooting. The guide distinguishes runnable experiments from historical or proposed work.

1. **Current proposal:** [Team07_T14_proposal.pptx](output/Team07_T14_proposal.pptx).
2. **Experiment overview:** [TEAMMATE_START_HERE.md](TEAMMATE_START_HERE.md).
3. **Completed experiment map:** [ALL_EXPERIMENTS_PROPOSAL_MAP.md](research_archive/2026-10-11/ALL_EXPERIMENTS_PROPOSAL_MAP.md).
4. **Latest dated archive:** [2026-10-11 README](research_archive/2026-10-11/README.md).

## Current research direction

Keep OpenAD frozen and compare three query strategies on identical point samples:

- Canonical action word.
- One fixed paraphrase per action.
- Average of three fixed paraphrase score maps.

We aim to identify whether localization can improve without increasing false alarms. Report the measured trade-off even when no strategy improves both. Use 18-action mean IoU, per-action results, and false alarms on verified negative object-query pairs. Keep prompts and the decision rule fixed before evaluation, and calculate paired uncertainty by object.

The proposed three-condition study is not a completed result. The current proposal predicts no reliable positive mean-IoU gain from averaging. Earlier files discuss different candidate ideas, including background-score adjustment and five-prompt embedding averaging. They remain historical drafts; use the current proposal above to identify the agreed three-score-map comparison. Later methodology discussions must be reconciled explicitly before running the study.

## What the existing evidence supports

- E17 compares training-associated query names with replacement names. It measures vocabulary sensitivity, not proof of new-action generalization.
- E18 tests fixed prompt transfer to OpenAD. Effects vary by action, and a localization gain can coincide with more false alarms.
- E08-E16 include exploratory studies with an unofficial Espresso-3D implementation and Affogato-pretrained weights. They are not official Affogato benchmark reproductions.
- Affostruction inference was not completed locally. No local Affostruction performance claim is supported.

Use the dated result reports for exact metrics, object manifests, devices and limitations. Earlier pilot notes describe the state at the time they were written and can be superseded by later runs.

## Repository contents

| Location | Contents |
| --- | --- |
| `pilot/` | Experiment runners, analysis scripts, result tables, saved predictions and figures |
| `research_archive/2026-10-05/` | Earlier OpenAD baseline and diagnostic evidence, including adapted source snapshots |
| `research_archive/2026-10-08/` | Query-sensitivity studies |
| `research_archive/2026-10-11/` | Later prompt, calibration, transfer and CPU/MPS comparison records |
| `output/` | Current Team 07 proposal |
| `pilot_scores*/`, `partial_dataset_scores/` | Dataset-analysis summaries |
| `MPS_RERUN_GUIDE.md` | Existing instructions for E17/E18 on Apple MPS |

The repository retains small raw prediction arrays and archived run records. Large model downloads, Python environments, separate upstream Git checkouts, duplicate ZIP packages and temporary presentation renders are excluded. Superseded proposal output variants and paper-presentation preparation remain local and are excluded from the current repository tree.

## Reproducing experiments on another computer

This is a research handoff, not a turnkey installation. Several scripts and run records contain the original author's absolute paths. Do not expect them to run unchanged on a collaborator's computer.

1. Obtain the external source, weights and dataset separately, following their upstream licenses and instructions.
2. Read the relevant experiment's `run.json`, `PROTOCOL.md` and `RESULTS.md` before running anything. Some experiments provide only a subset of these files.
3. Update executable paths for your machine. In particular, many scripts reference an external OpenAD checkout under `/Users/yaminmyataung/`. Preserve archived JSON/log provenance rather than rewriting past run records.
4. Match the checkpoint identity, label/query ordering, point sampling and metric implementation. Archived adapted source files are not a full replacement for setting up the original repository.
5. Use a new output directory for a new run. Keep CPU and MPS outputs separate. Start with the small smoke runs in the MPS guide if using the same Mac environment.
6. Never select prompts or thresholds using the final evaluation results. Disclose reuse of previously examined validation data as exploratory.

No new model inference or full experiment rerun was performed as part of this GitHub handoff.

## Collaboration

Clone this private repository using an account that already has access:

```sh
git clone https://github.com/Yamin310702/T14-Affordance-Project.git
cd T14-Affordance-Project
git switch -c yourname/research-update
```

Keep research changes on a separate branch and open a pull request for review. Record new experiment protocols and results together, including failures. Please keep model checkpoints and complete datasets out of ordinary Git commits.

If a collaborator cannot clone the private repository, the owner must grant that GitHub account access under the repository's collaborator settings. Uploading files does not automatically invite collaborators.

## Upstream references and attribution

- [OpenAD paper](https://arxiv.org/abs/2303.02401) and [official source](https://github.com/Fsoft-AIC/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds).
- [Affogato project](https://junha-l.github.io/affogato/) and [unofficial Espresso implementation](https://github.com/LHC0312/espresso-3d-reimplementation).
- [Affostruction source](https://github.com/chrockey/Affostruction).

Archived third-party source retains its included license files. Papers and course materials retain their original authorship and terms. This handoff does not relicense those materials.
