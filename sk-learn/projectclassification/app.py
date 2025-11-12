import pandas as pd
import numpy as np
import joblib
import pickle
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score,StratifiedShuffleSplit



MODEL = "model.pkl"
PIPELINE = "pipeline.pkl"

if not os.path.exists(MODEL):
    data = pd.read_csv("custommers.csv")
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')
    data["change_cat"] = pd.cut(data["MonthlyCharges"], bins=[0,20,40,60,80,100,np.inf], labels=[1,2,3,4,5,6])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_ix, test_ix in split.split(data, data["change_cat"]):
        trainmixdata = data.loc[train_ix]
        testmixdata = data.loc[test_ix]
            
    testmixdata.to_csv("input.csv", index = False)
    traindata = trainmixdata.drop(columns = ["customerID", "Churn", "change_cat"])
        
    num_att = traindata.select_dtypes(include = "number").columns.tolist()
    cat_att = [col for col in traindata.columns if col not in num_att]

    def preprocessing(num_att, cat_att):
        num = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scalling", StandardScaler())
        ])
        cat = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoding", OneHotEncoder(handle_unknown="ignore"))
        ])
        combine = ColumnTransformer([
            ("num", num, num_att),
            ("cat", cat, cat_att)
        ])
        return combine
    pipcall = preprocessing(num_att, cat_att)
    encoder = OrdinalEncoder()
    x_train = pipcall.fit_transform(traindata)
    y_train = encoder.fit_transform(trainmixdata["Churn"].values.reshape(-1,1)).ravel()
    # print(pd.DataFrame(processtraindt))


    model = GradientBoostingClassifier()
    model.fit(x_train, y_train )
        

    joblib.dump(model, MODEL)
    joblib.dump(pipcall, PIPELINE)
    print("model saved !!")
else:
    model = joblib.load(MODEL)
    pipcall = joblib.load(PIPELINE)
    data = pd.read_csv("input.csv")
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')
    testx = data.drop(columns = ["customerID", "Churn", "change_cat"])
     
    encoder = OrdinalEncoder()
    x_test = pipcall.fit_transform(testx)
    # y_test = encoder.fit_transform(data["Churn"].values.reshape(-1,1)).ravel()
    pred = model.predict(x_test)
    # original_predictions = encoder.inverse_transform(pred)
    dfpred = pd.DataFrame(pred,columns=["pred"], index=testx.index)
    
    data["pred"] = dfpred["pred"].map(lambda x: "Yes" if x == 1 else "No")
# Step 3: Flatten the result for easier use
    data.to_csv("output.csv", index=False)
    print(testx)
    print(data)
    print("model exist !!")


