import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def ordinal_encoder(input_val,feats):
    len1=len(feats)
    feat_val=list(1+np.arange(len1))
    feat_key=feats
    feat_dict=dict(zip(feat_key,feat_val))
    value=feat_dict[input_val]
    return value
def get_prediction(data,model):
    return model.predict(data)

