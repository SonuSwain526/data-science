import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error  
from sklearn.svm import SVR

data = pd.read_csv("housing.csv")
# print(data.head())
data["incom_Cat"] = pd.cut(data["median_income"], bins=[0,1.5,3.0,4.5,6.0, np.Infinity], labels=[1,2,3,4,5])
def dividing(data, pct):
    strata = StratifiedShuffleSplit(n_splits=1, test_size=pct, random_state=42)
    for train_ix, test_ix in strata.split(data, data["incom_Cat"]):
        stratified_Train = data.loc[train_ix]
        stratified_Test = data.loc[test_ix]
        return stratified_Train
    
trainAll = dividing(data, 0.3)

trainset = trainAll.drop(["median_house_value", "incom_Cat"], axis=1)
lavel = trainAll["median_house_value"]

try:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scalling", MinMaxScaler())
    ])

    cat_piplene = Pipeline([
        # ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encodig", OneHotEncoder(handle_unknown="ignore"))
    ])

    combine = ColumnTransformer([
        ("num", num_pipeline, trainset.drop(["ocean_proximity"], axis = 1).columns),
        ("cat", cat_piplene, trainset.columns[8:9])
    ])

    final_Trainset = combine.fit_transform(trainset)

except :
    print("something!!", )


# # # testing which model is worth it
# lin_estimator = LinearRegression()
# lin_estimator.fit(final_Trainset, lavel)
# lin_prdict = lin_estimator.predict(final_Trainset)
# # # print(mean_squared_error(lin_prdict, lavel, squared=False))

# dectree_estimator = DecisionTreeRegressor()
# dectree_estimator.fit(final_Trainset, lavel)
# dectree_pred = dectree_estimator.predict(final_Trainset)
# # print(mean_squared_error(dectree_pred, lavel, squared=False))


# randomforest_estimator = RandomForestRegressor()
# randomforest_estimator.fit(final_Trainset, lavel)
# randomforest_predict = randomforest_estimator.predict(final_Trainset)
# # # print(mean_squared_error(randomforest_predict, lavel, squared=False))



# accurate_error_dectree = -cross_val_score(
#     dectree_estimator,
#     final_Trainset,
#     lavel,
#     scoring="neg_root_mean_squared_error",
#     cv=10
# )
# print(accurate_error_dectree)

# accurate_error_linreg = -cross_val_score(
#     lin_estimator,
#     final_Trainset,
#     lavel,
#     scoring="neg_root_mean_squared_error",
#     cv=10
# )
# print(accurate_error_linreg)
# accurate_error_radomforest = -cross_val_score(
#     randomforest_estimator,
#     final_Trainset,
#     lavel,
#     scoring="neg_root_mean_squared_error",
#     cv=10
# )
# print(accurate_error_radomforest)

svr = SVR()
svr.fit(trainsfet, lavel)
accurate_error_radomforest = -cross_val_score(
    svr,
    final_Trainset,
    lavel,
    scoring="neg_root_mean_squared_error",
    cv=10
)
print(accurate_error_radomforest)



# error = root_mean_squared_error() 