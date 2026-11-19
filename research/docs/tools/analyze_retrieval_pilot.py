#!/usr/bin/env python3
"""Read-only independent math, prediction, source and budget verification."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'docs/tools'))
from run_rgb_pilot import verify_run, sha, inside, require

IDENTITY=('index','gloss','class_id','signer_id','split','source_path','video_sha256')
PREREG='docs/environment/retrieval-preregistration-20260926.json'
OLD='runs/20260923T050910Z-official-val-v2-evaluation-019ecd54/evaluation/predictions.jsonl'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def normalize(x):
    x=np.asarray(x,dtype=np.float64)
    n=np.sqrt(np.sum(x*x,axis=1,keepdims=True))
    require(np.isfinite(x).all() and np.isfinite(n).all() and bool((n>1e-12).all()),'Bad CLS')
    return x/n


def reference_scores(train, labels, signers, queries, candidates, logits):
    """Query-at-a-time math, independent of production grouping/chunk helpers."""
    train,queries=normalize(train),normalize(queries)
    groups={}
    for label in np.unique(labels):
        groups[int(label)]=[np.flatnonzero((labels==label)&(signers==signer))
                           for signer in sorted(set(signers[labels==label]))]
    kernel=np.empty(candidates.shape,np.float64)
    for i,query in enumerate(queries):
        for j,label in enumerate(candidates[i]):
            require(int(label) in groups,'Candidate has no train support')
            means=[]
            for ix in groups[int(label)]:
                cosine=np.clip(np.sum(train[ix]*query,axis=1),-1.,1.)
                means.append(float(np.exp(-10.*(1.-cosine)).mean()))
            kernel[i,j]=np.mean(means)
    memory=kernel/kernel.sum(axis=1,keepdims=True)
    shifted=logits.astype(np.float64)-logits.max(axis=1,keepdims=True).astype(np.float64)
    base=np.exp(shifted);base/=base.sum(axis=1,keepdims=True)
    blend=.75*base+.25*memory
    return {'memory_scores':kernel,'base_probabilities':base,'memory_probabilities':memory,'blend_probabilities':blend}


def order(candidates,scores):
    return np.take_along_axis(candidates,np.argsort(-scores,axis=1,kind='stable'),axis=1)


def metrics(targets,predictions):
    hit1=predictions[:,0]==targets;hit5=(predictions==targets[:,None]).any(axis=1)
    counts=np.bincount(targets,minlength=2731)
    correct1=np.bincount(targets,weights=hit1,minlength=2731)
    correct5=np.bincount(targets,weights=hit5,minlength=2731)
    present=counts>0
    return {'samples':len(targets),'failed':0,'correct_top1':int(hit1.sum()),'correct_top5':int(hit5.sum()),
            'top1':float(hit1.mean()),'top5':float(hit5.mean()),
            'macro_top1':float((correct1[present]/counts[present]).mean()),
            'macro_top5':float((correct5[present]/counts[present]).mean())}


def paired(targets,base,changed):
    a,b=base[:,0]==targets,changed[:,0]==targets
    return {'samples':len(targets),'both_correct':int((a&b).sum()),'fixed':int((~a&b).sum()),
            'regressed':int((a&~b).sum()),'both_wrong':int((~a&~b).sum()),
            'net_correct':int(b.sum()-a.sum()),'changed_top1':int((base[:,0]!=changed[:,0]).sum())}


def load_feature_run(run,split,prereg):
    run=inside(run);proof=verify_run(run); directory=run/'features';manifest=read(directory/'manifest.json')
    require(manifest['status']=='completed' and manifest['scope']=='formal' and manifest['weights_unchanged']
            and manifest['test_used'] is False and manifest['optimizer_steps']==0,'Wrong feature scope/state')
    require(manifest['checkpoint_sha256']==prereg['model']['checkpoint_sha256'],'Checkpoint differs')
    require(set(manifest['splits'])=={split},'Feature split differs')
    entry=manifest['splits'][split]
    require(entry['samples']==prereg['data'][split+'_count'] and entry['failed']==0,'Feature population differs')
    for name,expected in manifest['source_sha256'].items():
        require(sha(inside(name))==expected,'Frozen extraction input changed: '+name)
    metadata=inside(entry['metadata_path'],directory,relative=True)
    featurefile=inside(entry['features_path'],directory,relative=True)
    require(sha(metadata)==entry['metadata_sha256'] and sha(featurefile)==entry['features_sha256'],'Feature artifact SHA differs')
    records=rows(metadata)
    with np.load(featurefile,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    source=read(inside(prereg['data'][split+'_cache_manifest']))
    require(sha(inside(prereg['data'][split+'_cache_manifest']))==prereg['data'][split+'_cache_sha256'],'Cache index changed')
    require(len(records)==len(source['samples'])==entry['samples'],'Missing or extra rows')
    for n,(record,item) in enumerate(zip(records,source['samples'])):
        require(record['row']==n and record['identity']==item['identity'] and all(record[k]==item[k] for k in IDENTITY if k!='class_id'),'Feature identity/order differs')
        require(manifest['label_map'][str(record['class_id'])]==record['gloss'],'Feature label map mismatch')
        require(record['features_sha256']==item['features_sha256'] and record['pose_sha256']==item['pose_sha256'],'Source payload differs')
    return records,data,manifest,proof


def check_summary_contract(summary):
    require(summary.get('status')=='completed' and summary.get('scope')=='retrieval_pilot_full2731'
            and summary.get('failed')==0 and summary.get('optimizer_steps')==0
            and summary.get('test_used') is False,'Wrong rerank scope/state')
    require(summary.get('train_samples')==40154 and summary.get('val_samples')==10304
            and summary.get('candidate_classes')==2731 and summary.get('topk')==5,'Wrong rerank population')


def check_pair_details(stored_pair,expected_rows):
    for kind,expected in expected_rows.items():
        require(stored_pair[kind]==expected,'Paired example metadata differs: '+kind)


def analyze(args):
    prereg_path=inside(PREREG);prereg=read(prereg_path)
    require(prereg['registered_before_retrieval_scores'] is True and prereg['optimizer_steps']==0
            and prereg['retrieval']['beta']==10. and prereg['retrieval']['blend_memory_weight']==.25
            and prereg['primary_method']=='fixed_blend' and prereg['test_used'] is False,'Registered recipe differs')
    launch_path=inside(args.launch);launch=read(launch_path);state=read(launch_path.parent/'state.json')
    require(state['state']=='completed','Lane has not completed')
    tracked={str(prereg_path.relative_to(ROOT)):sha(prereg_path),str(launch_path.relative_to(ROOT)):sha(launch_path),
             str((launch_path.parent/'state.json').relative_to(ROOT)):sha(launch_path.parent/'state.json')}
    require(launch['prereg_sha256']==tracked[PREREG],'Prereg differs from launch')
    for name,expected in launch['frozen_files'].items():
        require(sha(inside(name))==expected,'Frozen launch input changed: '+name)
    for run in (inside(args.train_run),inside(args.val_run),inside(args.rerank_run)):
        for leaf in ('manifest.json','artifacts.json','config.json'):
            tracked[str((run/leaf).relative_to(ROOT))]=sha(run/leaf)
    tracked[OLD]=sha(inside(OLD))
    train,tx,tm,tp=load_feature_run(args.train_run,'train',prereg)
    val,vx,vm,vp=load_feature_run(args.val_run,'val',prereg)
    require(not ({x['video_sha256'] for x in train}&{x['video_sha256'] for x in val})
            and not ({x['signer_id'] for x in train}&{x['signer_id'] for x in val}),'Memory/query overlap')
    rerank_run=inside(args.rerank_run);rp=verify_run(rerank_run);out=rerank_run/'rerank'
    summary=read(out/'summary.json')
    check_summary_contract(summary)
    cfg=read(out/'config.json')
    require(cfg['beta']==10 and cfg['memory_weight']==.25 and cfg['base_weight']==.75
            and cfg['top_k']==5 and not cfg['engineering_smoke'] and cfg['primary_method']=='fixed_blend'
            and cfg['train_manifest_sha256']==sha(inside(args.train_run)/'features/manifest.json')
            and cfg['val_manifest_sha256']==sha(inside(args.val_run)/'features/manifest.json'),
            'Rerank input/config differs from fixed extraction')
    for mode,key in (('train_extraction','train_run'),('val_extraction','val_run'),('rerank','rerank_run')):
        require(state['stages'][mode]['state']=='completed'
                and inside(state['stages'][mode]['run_dir'])==inside(getattr(args,key)),
                'Analysis inputs differ from completed lane')
    sorted_ix=sorted(range(len(train)),key=lambda i:train[i]['video_sha256'])
    ix=np.asarray(sorted_ix)
    reference=reference_scores(tx['embeddings'][ix],np.array([train[i]['class_id'] for i in ix]),
                               np.array([train[i]['signer_id'] for i in ix]),vx['embeddings'],
                               vx['top5_class_ids'],vx['top5_logits'])
    expected={'original':vx['top5_class_ids'],
              'pure_memory':order(vx['top5_class_ids'],reference['memory_probabilities']),
              'fixed_blend':order(vx['top5_class_ids'],reference['blend_probabilities'])}
    targets=np.array([x['class_id'] for x in val]); all_metrics={k:metrics(targets,v) for k,v in expected.items()}
    for name,pred in expected.items():
        require(np.array_equal(np.sort(pred,axis=1),np.sort(expected['original'],axis=1)),'Candidate set changed')
        require(np.array_equal((pred==targets[:,None]).any(axis=1),
                               (expected['original']==targets[:,None]).any(axis=1)),'Per-video Top5 changed')
    old=rows(inside(OLD));require(len(old)==len(val),'Old baseline population differs')
    for n,(before,now) in enumerate(zip(old,val)):
        require(all(before[k]==now[k] for k in IDENTITY),'Old official reference identity differs')
        require([x['class_id'] for x in before['top5']]==expected['original'][n].tolist(),'Official Top5 rank differs')
    pairs={name:paired(targets,expected['original'],expected[name]) for name in ('pure_memory','fixed_blend')}
    by_signer={signer:{name:metrics(targets[ixs],p[ixs]) for name,p in expected.items()}
               for signer in sorted({x['signer_id'] for x in val})
               for ixs in [np.array([i for i,x in enumerate(val) if x['signer_id']==signer])]} 
    margin=reference['base_probabilities'][:,0]-reference['base_probabilities'][:,1]
    locked=margin>1./3.+1e-12
    require(np.array_equal(expected['original'][locked,0],expected['fixed_blend'][locked,0]),'Blend violated margin bound')
    gain=pairs['fixed_blend']['net_correct'];macro=all_metrics['fixed_blend']['macro_top1']-all_metrics['original']['macro_top1']
    signal=gain*100>=.5*len(val) and macro>0
    result={'status':'verified','created_at_utc':datetime.now(timezone.utc).isoformat(),'test_used':False,'optimizer_steps':0,
            'metrics':all_metrics,'paired':pairs,'by_signer_descriptive_only':by_signer,
            'continuation_signal':{'pass':signal,'minimum_net_correct':52,'net_correct':gain,
                                   'macro_gain_percentage_points':100*macro,'auto_learned_reranker':False},
            'margin_bound':{'locked_samples':int(locked.sum()),'potentially_changeable_samples':int((~locked).sum()),
                            'baseline_errors_among_changeable':int(((~locked)&(expected['original'][:,0]!=targets)).sum()),
                            'note':'Conservative algebraic bound, not achieved correctable error count.'},
            'verification':{'original_all_top5_ranks_match_historical_official':True,'frozen_files':len(launch['frozen_files']),
                            'memory_query_video_and_signer_disjoint':True,'all_per_video_top5_sets_and_hits_unchanged':True},
            'artifact_verification':{'train':tp,'val':vp,'rerank':rp},'input_sha256':tracked,
            'runs':{'train':str(inside(args.train_run).relative_to(ROOT)),'val':str(inside(args.val_run).relative_to(ROOT)),
                    'rerank':str(rerank_run.relative_to(ROOT))}}
    names={'memory_scores':'memory_kernel_scores','base_probabilities':'base_probability',
           'memory_probabilities':'memory_probability','blend_probabilities':'blend_probability'}
    max_error={}
    with np.load(out/'scores.npz',allow_pickle=False) as z:
        require(set(z.files)==set(names.values())|{'candidate_ids'},'Unexpected score array inventory')
        require(np.array_equal(z['candidate_ids'],vx['top5_class_ids']),'Scored candidates differ')
        for key,name in names.items():
            require(z[name].dtype==np.float64 and z[name].shape==reference[key].shape
                    and np.isfinite(z[name]).all() and np.allclose(z[name],reference[key],rtol=1e-12,atol=1e-14),
                    'Independent retrieval math differs: '+name)
            max_error[name]=float(np.abs(z[name]-reference[key]).max())
    for mode,pred in expected.items():
        persisted=rows(out/mode/'predictions.jsonl')
        require(len(persisted)==len(val),'Prediction row count differs')
        for n,(got,item) in enumerate(zip(persisted,val)):
            require(got['row']==n and got['index']==item['index'] and got['identity']==item['identity']
                    and got['split']=='val' and got['video_sha256']==item['video_sha256']
                    and got['signer_id']==item['signer_id'] and got['target']==item['class_id']
                    and got['gloss']==item['gloss'] and got['method']==mode,'Prediction identity differs')
            require(got['original_candidate_ids']==vx['top5_class_ids'][n].tolist()
                    and got['original_logits']==vx['top5_logits'][n].tolist()
                    and got['ranked_class_ids']==pred[n].tolist() and got['prediction']==int(pred[n,0]),
                    'Prediction ranking differs from independent math')
            require(got['correct_top1']==bool(pred[n,0]==targets[n])
                    and got['correct_top5']==bool((pred[n]==targets[n]).any()),'Prediction correctness differs')
            require(got['ranking_positions']==[got['original_candidate_ids'].index(int(c)) for c in pred[n]],
                    'Saved ranking positions differ')
            for ref,key in names.items():
                require(np.allclose(got[key],reference[ref][n],rtol=1e-12,atol=1e-14),'Saved row scores differ')
        stored=read(out/mode/'metrics.json')
        require(stored==summary['metrics'][mode],'Metric summary copies differ')
        require(stored['candidate_classes']==stored['observed_classes']==2731
                and set(stored['per_class'])=={str(i) for i in range(2731)},'Metric class inventory differs')
        require(read(out/mode/'by_signer.json')==summary['by_signer'][mode],'Signer sidecar differs')
        for key,value in all_metrics[mode].items():
            require(abs(stored[key]-value)<1e-14,'Aggregate metric differs: '+key)
        for label in range(2731):
            mask=targets==label;m=metrics(targets[mask],pred[mask]);entry=stored['per_class'][str(label)]
            for key in ('samples','correct_top1','correct_top5','top1','top5'):
                require(abs(entry[key]-m[key])<1e-14,'Per-class metric differs')
        for signer,signer_metrics in by_signer.items():
            for key,value in signer_metrics[mode].items():
                require(abs(summary['by_signer'][mode][signer][key]-value)<1e-14,'Signer metric differs')
        if mode!='original':
            stored_pair=summary['paired_comparisons'][mode]
            require(read(out/mode/'paired_vs_original.json')==stored_pair,'Paired sidecar differs')
            require(stored_pair['fixed_count']==pairs[mode]['fixed']
                    and stored_pair['regressed_count']==pairs[mode]['regressed']
                    and stored_pair['net_correct']==pairs[mode]['net_correct']
                    and stored_pair['changed_top1_count']==pairs[mode]['changed_top1'],'Paired counts differ')
            b=expected['original'][:,0]==targets;c=pred[:,0]==targets
            for kind,mask in (('fixed',~b&c),('regressed',b&~c),('changed_top1',expected['original'][:,0]!=pred[:,0])):
                details=[{'row':int(i),'index':val[i]['index'],'identity':val[i]['identity'],
                          'video_sha256':val[i]['video_sha256'],'signer_id':val[i]['signer_id'],
                          'target':int(targets[i]),'original':int(expected['original'][i,0]),
                          'candidate':int(pred[i,0])} for i in np.flatnonzero(mask)]
                check_pair_details(stored_pair,{kind:details})
    require(summary['continuation_signal']['pilot_signal'] is bool(signal),'Primary stopping signal differs')
    result['verification'].update(independent_kernel_probability_max_absolute_error=max_error,
                                  all_30912_prediction_rows_exact=True,all_class_and_signer_metrics_recomputed=True)
    for name,expected_sha in launch['frozen_files'].items():
        require(sha(inside(name))==expected_sha,'Frozen launch changed during verification: '+name)
    for name,expected_sha in tracked.items():
        require(sha(inside(name))==expected_sha,'Input changed during verification: '+name)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('launch','train-run','val-run','rerank-run','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();output=inside(a.output);require(not output.exists(),'Analysis output is immutable')
    result=analyze(a)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'status':result['status'],'signal':result['continuation_signal']['pass']}))

if __name__=='__main__':main()
