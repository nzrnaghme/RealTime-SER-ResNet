"""Pre-test bounded SVM head extension on fixed emotion2vec embeddings.

Authorized after the logistic validation results and before any test inference.
All three candidates are retained. No test predictions or test metrics.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from emotion2vec_baseline_v10 import CLASSES, REVISION, CHECKPOINT_SHA, rows_from, sha, save_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--features',required=True);p.add_argument('--out',required=True)
    args=p.parse_args()
    rows=rows_from(args.manifest)
    side=Path(args.features).with_suffix('.json')
    meta=json.loads(side.read_text())
    assert meta['npz_sha256']==sha(args.features) and meta['manifest_sha256']==sha(args.manifest)
    assert meta['model_revision']==REVISION and meta['checkpoint_sha256']==CHECKPOINT_SHA
    assert meta['upstream_parameters_updated'] is False
    with np.load(args.features,allow_pickle=False) as f:
        assert f['sample_ids'].tolist()==[r['sample_id'] for r in rows]
        x=f['embeddings'].copy()
    assert x.shape==(len(rows),768) and np.isfinite(x).all()
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    val=np.array([i for i,r in enumerate(rows) if r['split']=='val'])
    ytrain=np.array([int(rows[i]['y']) for i in train]);yval=np.array([int(rows[i]['y']) for i in val])
    corpora=sorted({rows[i]['corpus'] for i in train});weights=np.empty(len(train))
    for corpus in corpora:
        for k in range(4):
            mask=np.array([rows[i]['corpus']==corpus for i in train])&(ytrain==k)
            assert mask.sum()>0
            weights[mask]=len(train)/(len(corpora)*4*mask.sum())
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'selection.json').exists(), 'Preserve existing selected experiment'
    config={'head_family':'rbf_svm','C_grid':[0.1,1.0,10.0],'kernel':'rbf','gamma':'scale',
            'probability':False,'random_state':17,'labels':CLASSES,'class_weight':None,
            'scaler_fit':'train only, unweighted','sample_weights':'N_train/(number_of_corpora*4*training_count_in_corpus_class)',
            'selection':'highest validation UAR; smaller C on exact tie','n_train':len(train),'n_val':len(val),
            'features_sha256':sha(args.features),'feature_provenance_sha256':sha(side),'manifest_sha256':sha(args.manifest),
            'script_sha256':sha(__file__),'upstream_revision':REVISION,'upstream_sha256':CHECKPOINT_SHA,
            'test_predictions_computed':False,'protocol_extension':'One RBF SVM family added after logistic validation results and before any test inference; same fixed C grid, no other feature or gamma search.'}
    save_json(out/'config.json',config);trials=[]
    for c in config['C_grid']:
        model=Pipeline([('scaler',StandardScaler()),('classifier',SVC(C=c,kernel='rbf',gamma='scale',probability=False,random_state=17))])
        t=time.perf_counter();model.fit(x[train],ytrain,classifier__sample_weight=weights)
        pred=model.predict(x[val]);scores=model.decision_function(x[val]);assert model.classes_.tolist()==[0,1,2,3]
        assert np.isfinite(scores).all()
        trial={'C':c,'validation_uar':float(recall_score(yval,pred,average='macro',labels=range(4),zero_division=0)),
               'validation_accuracy':float(accuracy_score(yval,pred)),'validation_macro_f1':float(f1_score(yval,pred,average='macro',labels=range(4),zero_division=0)),
               'validation_confusion_matrix':confusion_matrix(yval,pred,labels=range(4)).tolist(),
               'fit_and_validation_seconds':time.perf_counter()-t,'converged':model.named_steps['classifier'].fit_status_==0}
        assert trial['converged']
        stem=f'C{c:g}';joblib.dump(model,out/f'{stem}.joblib')
        with open(out/f'{stem}_validation_predictions.csv','w',newline='') as f:
            fields=['sample_id','corpus','speaker_id','y_true','y_pred']+[f'decision_{k}' for k in CLASSES]
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for j,i in enumerate(val):
                record={k:rows[i][k] for k in ['sample_id','corpus','speaker_id']}
                record.update(y_true=int(yval[j]),y_pred=int(pred[j]))
                record.update({f'decision_{k}':float(scores[j,z]) for z,k in enumerate(CLASSES)})
                writer.writerow(record)
        trial['model_sha256']=sha(out/f'{stem}.joblib');trials.append(trial)
        save_json(out/'validation_trials.json',trials);print(json.dumps(trial),flush=True)
    chosen=sorted(trials,key=lambda r:(-r['validation_uar'],r['C']))[0].copy()
    chosen.update(selected_model=f"C{chosen['C']:g}.joblib",test_predictions_computed=False,
                  config_sha256=sha(out/'config.json'),classes=CLASSES,head_family='rbf_svm')
    save_json(out/'selection.json',chosen)

if __name__=='__main__':main()
