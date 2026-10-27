from pathlib import Path
import shutil, json, hashlib, subprocess, csv
w=Path(__file__).resolve().parents[1]
r=Path('/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds')
a=w/'research_archive'/'2026-10-05';a.mkdir(exist_ok=True)
records=[]
def copy(src,dest):
    src=Path(src);dest=a/dest
    if src.is_dir():
        for f in sorted(src.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts and '.git' not in f.parts:
                copy(f,dest.relative_to(a)/f.relative_to(src))
        return
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert h(src)==h(dest)
    records.append(dict(file=str(dest.relative_to(a)),source=str(src),sha256=h(dest),bytes=dest.stat().st_size))
for setting in ['full_shape','partial_view']:
    copy(r/f'{setting}_baseline_result.log',f'E01_openad_original_baselines/{setting}_baseline_result.log')
copy(r/'failure_visualizations','E02_openad_selected_failures/figures')
copy(r/'analyze_failure_scores.py','E02_openad_selected_failures/analyze_failure_scores.py')
copy(r/'visualize_failure.py','E02_openad_selected_failures/visualize_failure.py')
copy('/Users/yaminmyataung/.codex/attachments/cd9bb815-f0bb-4cf3-a7d5-1f1af39e1384/Pasted text.txt','E02_openad_selected_failures/user_reported_progress.txt')
for src,dst in [('pilot_scores','E03_openad_cpu_check'),('pilot_scores_mps','E04_openad_mps_check'),('partial_dataset_scores','E05_openad_dataset_analysis'),('pilot/espresso_real','E07_espresso_pretrained_single_case')]:
    copy(w/src,dst)
copy(w/'partial_dataset_analysis.log','E05_openad_dataset_analysis/run.log')
copy(w/'analyze_dataset_scores.py','code/analyze_dataset_scores.py')
for name in ['test_espresso_device.py','espresso_cpu_check.json','espresso_mps_check.json','ESPRESSO_MAC_AUDIT.md']:
    copy(w/'pilot'/name,'E06_espresso_compatibility/'+name)
for name in ['run_espresso_real.py','plot_espresso_real.py','espresso_environment.txt','espresso_real.log']:
    copy(w/'pilot'/name,'E07_espresso_pretrained_single_case/'+name)
for name in ['README.md','pilot_manifest.json','prepare_manifest.py']:
    copy(w/'pilot'/name,'P01_three_model_pilot_not_run/'+name)
copy(w/'T14_next_experiment.md','planning/T14_next_experiment.md')
for repo,folders in [(r,['models','dataset','utils','config']), (w/'pilot/espresso-3d-reimplementation',['espresso','partfield'])]:
    label='openad' if repo==r else 'espresso_unofficial'
    for folder in folders: copy(repo/folder,Path('code')/label/folder)
    copy(repo/'LICENSE',Path('code')/label/'LICENSE')
    for name,args in [('commit.txt',['rev-parse','HEAD']),('working_changes.patch',['diff']),('git_status.txt',['status','--short'])]:
        p=a/'code'/label/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(subprocess.check_output(['git','-C',str(repo),*args],text=True))
copy(w/'pilot/Affostruction/manifests','P01_three_model_pilot_not_run/official_manifests')
(a/'copied_file_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
rows=[
['E01','OpenAD original baselines','completed; existing logs','Full mIoU 0.144121; partial mIoU 0.124964','No rerun; original logs retained'],
['E02','OpenAD selected failures','completed; figures and user report','clothe/wrap/pour selected cases','Original score stdout not found; values in user report are not newly verified'],
['E03','OpenAD CPU one-sample check','completed','Sample 452','Compatibility/diagnostic check only'],
['E04','OpenAD MPS one-sample check','completed','Sample 452','Fixed-seed scores differ from original selected-case run'],
['E05','OpenAD partial dataset diagnostic','completed','9140 views; mIoU 0.1253635702','Batch 1; point-weighted across repeated views'],
['E06','Espresso component compatibility','completed','CPU encoder; MPS encoder and synthetic-text decoder','Random weights; not semantic evaluation'],
['E07','Unofficial Espresso pretrained inference','completed','Vase 452; 2048 scores; 1.643 s','One out-of-protocol case; no comparative accuracy claim'],
['P01','Three-model pilot / Affostruction','planned, not run','10 selected object IDs','No Affostruction predictions; pilot assets missing']]
with (a/'experiment_register.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['ID','experiment','status','result','limitation']);writer.writerows(rows)
print(f'Copied and hash-verified {len(records)} files into {a}')
