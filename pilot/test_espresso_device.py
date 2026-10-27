"""Synthetic encoder compatibility check. Random weights: NOT an accuracy experiment."""
import argparse,json,sys,time,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',default='cpu'); a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parent/'espresso-3d-reimplementation'))
import torch
from espresso.model import PartFieldEncoder, Espresso3D
result={'device':a.device,'torch':torch.__version__,'pretrained':False,'points':2048}
t=time.monotonic()
try:
    torch.manual_seed(1)
    enc=PartFieldEncoder(device='cpu').eval().to(a.device)
    result['parameters']=sum(p.numel() for p in enc.parameters())
    x=torch.rand(1,2048,3,device=a.device)*2-1
    with torch.inference_mode(): y=enc(x)
    result.update(shape=list(y.shape),finite=bool(torch.isfinite(y).all().item()),status='passed')
    del enc
    class SyntheticText(torch.nn.Module):
        width=1024
        embed_dim=768
        def forward(self,texts,device):
            return (torch.zeros(len(texts),768,device=device),
                    torch.zeros(len(texts),128,1024,device=device),
                    torch.zeros(len(texts),128,dtype=torch.bool,device=device))
    model=Espresso3D(text_encoder=SyntheticText(),device='cpu').eval().to(a.device)
    with torch.inference_mode(): logits=model(x,['synthetic query'],point_feat=y)
    assert logits.shape==(1,2048) and bool(torch.isfinite(logits).all().item())
    result['decoder_with_synthetic_text']='passed'
    result['real_text_encoder_tested']=False
    assert list(y.shape)==[1,2048,448] and result['finite']
except Exception as e:
    result.update(status='failed',error=str(e)); traceback.print_exc()
result['seconds']=time.monotonic()-t
(Path(__file__).resolve().parent/f'espresso_{a.device}_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
