import numpy as np
import pandas as pd
import joblib
import pickle
import os
import math

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL = "model.pkl"
PIPELINE = "pipeline.pkl"

if not (os.path.exists(MODEL)):
    data = pd.read_csv("housing.csv")

    def devidedata(data, pct):
        data["income_Cat"] = pd.cut(data["median_income"], bins=[0, 1.5, 3.0, 4.5, 6.0, np.Infinity], labels=[1,2,3,4,5])
        split = StratifiedShuffleSplit(n_splits=1, test_size=pct,random_state=42)

        for train_ix, test_ix in split.split(data, data["income_Cat"]):
            mixtrain = data.loc[train_ix]
            mixtest = data.loc[test_ix]
            return mixtrain, mixtest
    trainngset, testset = devidedata(data, 0.2)
    
    modifie_train = trainngset.copy()

    trainset_features = modifie_train.drop(["income_Cat","median_house_value"], axis = 1)
    trainset_label = modifie_train["median_house_value"]
    testset.to_csv("input1.csv", index=False)

    num_clms = trainset_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_clms = ["ocean_proximity"]

    def pipilingin(onlynumclm, onlycatclms):
        num = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scalling", StandardScaler())
        ])
        cat = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoding", OneHotEncoder(handle_unknown="ignore"))
        ])
        fullpipeline = ColumnTransformer([
            ("num", num, onlynumclm),
            ("cat", cat, onlycatclms)
        ])

        return fullpipeline
    pipeline_Setup = pipilingin(num_clms, cat_clms)
    train_pipelined = pipeline_Setup.fit_transform(trainset_features)
    print(train_pipelined)

    #train the model    
    # model = SVR()
    model = RandomForestRegressor()
    model.fit(train_pipelined, trainset_label)

    joblib.dump(model, MODEL)
    joblib.dump(pipeline_Setup, PIPELINE)

    print("model trained successfulyy")
else:
    # 1. Load model + pipeline
    model = joblib.load(MODEL)
    pipeline_Setup = joblib.load(PIPELINE)

    # 2. Read new input
    data = pd.read_csv("input1.csv").drop(["median_house_value", "income_Cat"], axis=1)

    # 3. Transform & Predict
    set_data = pipeline_Setup.transform(data)
    predictions = model.predict(set_data)

    # 4. Save output
    data["prediction"] = predictions
    # data.to_csv("output1.csv", "w",index=False)
    print("📄 Predictions saved to output.csv")

    mse = mean_squared_error(pd.read_csv("input1.csv")["median_house_value"], predictions)
    mae = mean_absolute_error(pd.read_csv("input1.csv")["median_house_value"], predictions)

    print(math.sqrt(mse), mae, r2_score(pd.read_csv("input1.csv")["median_house_value"], predictions))

