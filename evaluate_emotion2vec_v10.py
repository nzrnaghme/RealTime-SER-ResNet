"""Evaluate both frozen emotion2vec heads; no fit, tuning or model selection.

Requires a parent-authored global freeze with explicit hashes. Do not execute
until the experiment owner has closed all validation-based model choices.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from emotion2vec_baseline_v10 import CLASSES, rows_from, sha, save_json

def metrics(y,pred):
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=range(4),zero_division=0)
    return {'n':len(y),'accuracy':float(accuracy_score(y,pred)),
            'uar':float(recall.mean()),'macro_f1':float(f1.mean()),
            'weighted_f1':float(np.average(f1,weights=support)),
            'confusion_matrix_rows_true_columns_predicted':confusion_matrix(y,pred,labels=range(4)).tolist(),
            'classes':CLASSES,
            'per_class':{name:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i,name in enumerate(CLASSES)}}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--features',required=True);p.add_argument('--out',required=True)
    p.add_argument('--head-root',type=Path,help='Relocated artifact root containing the original head directory names; frozen hashes remain mandatory')
    args=p.parse_args()
    freeze=json.loads(Path(args.frozen).read_text())
    assert freeze['manifest_sha256']==sha(args.manifest),'Manifest not frozen'
    assert freeze['emotion2vec_features_sha256']==sha(args.features),'Features not frozen'
    entries=freeze['emotion2vec_heads']
    assert {e['head'] for e in entries}=={'logistic','rbf_svm'} and len(entries)==2
    rows=rows_from(args.manifest)
    with np.load(args.features,allow_pickle=False) as f:
        assert f['sample_ids'].tolist()==[r['sample_id'] for r in rows]
        x=f['embeddings'].copy()
    assert x.shape==(len(rows),768) and np.isfinite(x).all()
    test_ids=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    test_rows=[rows[i] for i in test_ids]
    ytest=np.array([int(r['y']) for r in test_rows])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'results.json').exists(),'Preserve prior test evaluation'
    # Validate every artifact before computing any predictions.
    checked=[]
    for entry in entries:
        frozen_model_path=Path(entry['model_path'])
        assert frozen_model_path.is_absolute(),'Frozen model paths must be absolute'
        model_path=(args.head_root.resolve()/frozen_model_path.parent.name/frozen_model_path.name
                    if args.head_root else frozen_model_path)
        config_path=model_path.parent/'config.json';selection_path=model_path.parent/'selection.json'
        assert sha(model_path)==entry['model_sha256']
        assert sha(config_path)==entry['config_sha256']
        assert sha(selection_path)==entry['selection_sha256']
        config=json.loads(config_path.read_text());selection=json.loads(selection_path.read_text())
        assert config['features_sha256']==sha(args.features) and config['manifest_sha256']==sha(args.manifest)
        assert config['feature_provenance_sha256']==sha(Path(args.features).with_suffix('.json'))
        assert selection['selected_model']==model_path.name and selection['model_sha256']==entry['model_sha256']
        assert selection['config_sha256']==entry['config_sha256'] and selection['classes']==CLASSES
        checked.append((entry,model_path,config,selection))
    records={}
    summary=[]
    for entry,model_path,config,selection in checked:
        head=entry['head'];model=joblib.load(model_path)
        classifier=model.named_steps['classifier']
        assert isinstance(classifier,LogisticRegression if head=='logistic' else SVC)
        assert model.classes_.tolist()==[0,1,2,3] and model.n_features_in_==768
        assert classifier.C==selection['C']
        start=time.perf_counter()
        pred=model.predict(x[test_ids])
        if head=='logistic':
            scores=model.predict_proba(x[test_ids]);score_kind='uncalibrated_logistic_probabilities'
        else:
            assert classifier.probability is False
            scores=model.decision_function(x[test_ids]);score_kind='svm_ovr_decision_values_not_probabilities'
        seconds=time.perf_counter()-start
        assert scores.shape==(len(test_ids),4) and np.isfinite(scores).all()
        with open(out/f'{head}_test_predictions.csv','w',newline='') as f:
            fields=['sample_id','corpus','speaker_id','split','y_true','y_pred','score_kind']+[f'score_{c}' for c in CLASSES]
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for i,row in enumerate(test_rows):
                record={k:row[k] for k in ['sample_id','corpus','speaker_id','split']}
                record.update(y_true=int(ytest[i]),y_pred=int(pred[i]),score_kind=score_kind)
                record.update({f'score_{c}':float(scores[i,j]) for j,c in enumerate(CLASSES)})
                writer.writerow(record)
        result={'pooled':metrics(ytest,pred),'corpus':{},'speaker':{},
                'validation_selection':selection,'frozen_entry':entry,'score_kind':score_kind,
                'cached_feature_prediction_seconds':seconds,'fitted_during_evaluation':False}
        for field,key in [('corpus','corpus'),('speaker_id','speaker')]:
            for group in sorted({r[field] for r in test_rows}):
                mask=np.array([r[field]==group for r in test_rows])
                result[key][group]=metrics(ytest[mask],pred[mask])
        result['corpus_balanced_uar']=float(np.mean([m['uar'] for m in result['corpus'].values()]))
        records[head]=result
        flat={'head':head,'C':selection['C'],'test_accuracy':result['pooled']['accuracy'],
              'test_uar':result['pooled']['uar'],'test_macro_f1':result['pooled']['macro_f1'],
              'test_corpus_balanced_uar':result['corpus_balanced_uar']}
        for corpus,m in result['corpus'].items():
            for name in ['accuracy','uar','macro_f1']:flat[f'{corpus}_{name}']=m[name]
        summary.append(flat)
        save_json(out/f'{head}_test_metrics.json',result)
    all_results={'frozen_file_sha256':sha(args.frozen),'manifest_sha256':sha(args.manifest),
                 'features_sha256':sha(args.features),'evaluation_script_sha256':sha(__file__),
                 'n_test':len(test_ids),'model_selection_from_test':False,'models':records,
                 'head_root_override':str(args.head_root.resolve()) if args.head_root else None,
                 'interpretation':'Single fixed speaker-disjoint split; this does not measure cross-corpus transfer. No direct gain claim versus the historical unverified 1685-item subset.'}
    save_json(out/'results.json',all_results)
    with open(out/'summary.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
