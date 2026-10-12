"""Reproducible mapping-only instance-selection A/B/C regression."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
from .instance_selection import candidate_request, predict, selection_result
from .realtime import Selector, consume_masks, consume_text, save_frame
from ..mapping import MapStore, SourceObservation
from ..mapping.semantic_frame import write_json
from ..schema import file_sha256


def load_batch(baseline,key):
    folder=baseline/'v8s-engine'/key
    data=json.loads((folder/'frame.json').read_text())
    instances=[]
    for i,inst in enumerate(data['instances']):
        instances.append({**{k:v for k,v in inst.items() if k!='texts'},'mask':np.load(folder/f'mask-{i:03}.npz')['mask']})
    batch=dict(source_image_id=data['source']['image_id'],rgb_hash=data['source']['rgb_hash'],
               instances=instances,provenance=data['provenance'],elapsed_s=data['segmentation_s'],
               possibly_truncated=len(instances)>=100)
    return data,batch


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=Path('/data/ticvla/outputs/landmark-realtime-20261012'))
    p.add_argument('--stage',choices=['qa','maps','freeze'],required=True)
    p.add_argument('--split',choices=['development','evaluation'],default='development')
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.baseline/'manifest.json').read_text());write_json(a.root/'manifest.json',manifest)
    rows=[r for r in manifest['rows'] if (r['variant']=='development')==(a.split=='development')]
    if a.stage=='freeze':
        paths=['ticvla/warehouse/perception/instance_selection.py','ticvla/warehouse/perception/realtime.py','ticvla/warehouse/mapping/store.py']
        write_json(a.root/'freeze.json',dict(code={p:file_sha256(p) for p in paths},model_profile=json.loads(Path('/data/ticvla/models/warehouse-realtime/selected.json').read_text()),
                   baseline=str(a.baseline),split='existing regression, not blind holdout',image_input='original + ID overlay, 448 each',max_new_tokens=2048))
        return
    if a.split=='evaluation':
        frozen=json.loads((a.root/'freeze.json').read_text())
        if any(file_sha256(k)!=v for k,v in frozen['code'].items()):raise ValueError('Frozen inference code changed')
    if a.stage=='qa':
        import torch
        torch.set_num_threads(2)
        model=Selector()
        for row in rows:
            folder=a.root/'qa'/row['key'];out=folder/'result.json'
            if out.exists():continue
            data,batch=load_batch(a.baseline,row['key']);source=data['source']
            request=candidate_request(source,batch,folder)
            result=predict(model,source,request);write_json(out,result)
            print(row['key'],result['valid'],result.get('parsed'),result.get('error'),result['elapsed_s'],flush=True)
        return
    for arm in ['A','B','C']:
        stores={}
        try:
            for row in rows:
                folder=a.root/arm/row['key']
                if (folder/'frame.json').exists():continue
                data,batch=load_batch(a.baseline,row['key'])
                store=stores.setdefault(row['variant'],None)
                if store is None:store=stores[row['variant']]=MapStore(a.root/arm/'maps'/row['variant'])
                source=store.add_observation(SourceObservation(**row['source']))
                if arm=='C':
                    request=json.loads((a.root/'qa'/row['key']/'request.json').read_text());batch['candidate_request']=request
                    qa=json.loads((a.root/'qa'/row['key']/'result.json').read_text());selection=selection_result(qa)
                else:
                    selection=copy.deepcopy(data['selection'])
                    if arm=='B' and set(selection['selected_categories'])&{'sign','label'}:
                        selection['selected_categories']=list(dict.fromkeys(selection['selected_categories']+['sign','label']))
                start=time.time()
                batch['prepared_projections']={i:store.prepare_projection(source['image_id'],x['mask']) for i,x in enumerate(batch['instances'])}
                admitted=consume_masks(store,source,batch,selection)
                ocr=json.loads((a.baseline/'ocr'/row['key']/'result.json').read_text())
                bound=consume_text(store,source,batch,admitted,ocr)
                save_frame(folder,source,batch,admitted,selection,ocr,bound)
                write_json(folder/'replay.json',dict(segmentation='saved real inference',ocr='saved real inference',map_processing_s=time.time()-start))
                print(arm,row['key'],sum(x['result']['status']=='accepted' for x in admitted),flush=True)
            for variant,s in stores.items():
                write_json(a.root/arm/'maps'/variant/'observed-landmarks.json',s.observed_landmarks())
        finally:
            for s in stores.values():
                if s:s.close()

if __name__=='__main__':main()
