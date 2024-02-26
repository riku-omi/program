import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import  datasets,metrics
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
import numpy as np
import statistics
from tqdm import tqdm
sk = pd.read_csv('___')
sk = sk.query('CAD==0 and stroke==0')


from sklearn .inspection import permutation_importance


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,confusion_matrix
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import (confusion_matrix, auc,roc_curve, recall_score,accuracy_score)


X=sk.drop(["p_stroke","p_CAD","normal","CAD",
          "stroke","all","TC",],axis=1)      





Y=(sk["normal"])
i=0
to=0

auc_=[]
acc=[]
recall=[]
spec=[]
ppv=[]#陽性的中率
npv=[]#陰性的中率
fpr_=[]
tpr_=[]



for n in tqdm(range(20)):
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,stratify=Y)
    lgbm=lgb.LGBMClassifier(objective='binary',
                  num_leaves=66,max_depth=6,max_bin=347,
                             min_child_samples=6,metric='auc',n_jobs=-1)
    

    usbc = BalancedBaggingClassifier(base_estimator=lgbm, n_jobs=-1, n_estimators=10,sampling_strategy='not minority')
    usbc.fit(X_train, y_train)

    
    pred=usbc.predict_proba(X_test)[:,1]
    print(pred)
    fpr,tpr,thres=roc_curve(y_test, pred)
    roc_auc=auc(fpr,tpr)
    #youden indexによるカットオフ値
    cutoff=tpr+1-fpr
    best_thres=np.argmax(cutoff)
    best=thres[best_thres]
    print(best)
    pred=np.where(pred>best,1,0)
    print(pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, pred).flatten()
    acc.append(accuracy_score(y_test, pred))
    recall.append(tp/(tp+fn))
    spec.append(tn / (tn + fp))
    auc_.append(roc_auc)



    
    #imp = pd.DataFrame(lgbm.feature_importances_,index=X.columns,columns=['importance'])
    #to=to+imp
    #average=to/20    
    
    ppv.append(tp/(tp+fp))
    npv.append(tn/(tn+fn))
    fpr_.append(fp/(fp+tn))
    tpr_.append(tp/(tp+fn))

#importance = average.sort_values('importance', ascending=True)
#print(importance)
#importance.to_csv('/Users/namboriku/Desktop/卒業研究/lgbm_imp_cad.csv') 
#lgb.plot_feature_importances_(lgbm, figsize=(10,10), max_num_features=70, importance_type='gain')

#imp=permutation_importance(usbc,X_train,y_train,n_repeats=20)
#imp_df=pd.DataFrame({"importances_mean":imp["importances_mean"],"importances_std":imp["importances_std"]},index=X.columns)
#imp_df = pd.DataFrame(zip(X.columns, imp["importances"].mean(axis=1)),columns=["Features","Importance"])
#imp_sort=imp_df.sort_values("Importance",ascending=(False))

#print(imp_sort)
#imp_sort.to_csv('/Users/namboriku/Desktop/卒業研究/lgbm_pimp_all2.csv') 


ave_acc=statistics.mean(acc)
ave_auc=statistics.mean(auc_)
ave_recall=statistics.mean(recall)
ave_spec=statistics.mean(spec)
ave_ppv=statistics.mean(ppv)
ave_npv=statistics.mean(npv)
ave_fpr=statistics.mean(fpr_)
ave_tpr=statistics.mean(tpr_)
#print(ave_fpr)
#print(ave_tpr)

plt.plot(fpr,tpr)
plt.fill_between(fpr, tpr, 0, alpha=0.1)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title('ROC curve')
plt.show()



print("\n")
print("ACC",ave_acc.round(decimals=3),"±",np.std(acc).round(decimals=3))
print("AUC",ave_auc.round(decimals=3),"±",np.std(auc_).round(decimals=3))
print("recall",ave_recall.round(decimals=3),"±",np.std(recall).round(decimals=3))
print("specificity",ave_spec.round(decimals=3),"±",np.std(spec).round(decimals=3))
print("ppv",ave_ppv.round(decimals=3),"±",np.std(ppv).round(decimals=3))
print("npv",ave_npv.round(decimals=3),"±",np.std(npv).round(decimals=3))
