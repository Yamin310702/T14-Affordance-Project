"""Freeze an exploratory pilot without selecting objects by model outcomes."""
from pathlib import Path
import hashlib, json, subprocess
root=Path(__file__).resolve().parent
source=root/'Affostruction/manifests/affogato_test.txt'
pairs=[line.strip().split(',') for line in source.read_text().splitlines() if line.strip()]
assert len(pairs)==1000 and len({x[1] for x in pairs})==1000
# Stable selection, independent of any labels, model outputs or asset availability.
ordered=sorted(pairs,key=lambda x:hashlib.sha256(('T14-pilot-v1:'+x[1]).encode()).hexdigest())
objects=[]
for sha,uid in ordered[:10]:
    objects.append(dict(objaverse_uid=uid,processed_sha256=sha,query_index=0,
        annotation=f'annotations/{uid}/xyzc.npy',query_file=f'annotations/{uid}/queries.json',
        rgbd_directory=f'renders_cond/{sha}',affordance_file=f'affordances/{sha}/affordance.npz',
        assets_status='not_downloaded',espresso_training_overlap='unchecked'))
result=dict(purpose='Exploratory development pilot; not final hypothesis test',
    selection='10 lowest SHA256(T14-pilot-v1:UID); no replacement based on results',
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'Affostruction',text=True).strip(),
    source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),objects=objects)
(root/'pilot_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print('Validated and selected 10 unique objects from 1,000 published test IDs.')
