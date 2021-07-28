#!/usr/bin/env python
# coding: utf-8

# In[1]:


import statistics as sta
import seaborn as sns
import matplotlib.pyplot as plt 


# In[2]:


import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz") 
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[3]:


fetch_housing_data()


# In[4]:


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[5]:


housing = load_housing_data()
housing.head()


# In[69]:


housing.info()


# In[70]:


housing['ocean_proximity'].value_counts()


# In[71]:


housing.describe()


# In[72]:


sta.median(housing['longitude']) 


# In[73]:


get_ipython().run_line_magic('matplotlib', 'inline')
housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[74]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[13]:


import numpy as np


# In[76]:


housing["income_cat"] = np.ceil(housing["median_income"] / 1.5) 
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# In[79]:


housing["median_income"].hist(bins=5,figsize=(6,6), width=0.7)


# In[83]:


housing["income_cat"].hist(bins=5,figsize=(6,6), width=0.3)


# In[84]:


from sklearn.model_selection import StratifiedShuffleSplit
    
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index] 
    strat_test_set = housing.loc[test_index]


# In[85]:


strat_test_set.head(5)


# In[86]:


strat_train_set.info()


# In[87]:


strat_train_set["income_cat"].hist(bins=5,figsize=(6,6), width=0.3)


# In[88]:


strat_test_set["income_cat"].hist(bins=5,figsize=(6,6), width=0.3)


# In[89]:


housing["income_cat"].value_counts() / len(housing)


# In[36]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# In[37]:


housing = strat_train_set.copy()


# In[39]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[40]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.3, s= housing["population"]/50, label="population",
            figsize=(10,7),c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True)


# In[41]:


corr_matrix = housing.corr()


# In[48]:


corr_matrix["median_house_value"].sort_values(ascending = False)


# In[49]:


from pandas.plotting import scatter_matrix


# In[50]:


attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[51]:


housing.plot(kind='scatter',x='median_income',y='median_house_value',alpha=0.1)


# In[52]:


housing.columns


# In[53]:


housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[54]:


corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)


# In[55]:


housing = strat_train_set.drop("median_house_value", axis=1) 
housing_labels = strat_train_set["median_house_value"].copy()


# In[56]:


median=housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median,inplace=True)


# In[57]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan , strategy='median')


# In[58]:


housing_num=housing.drop("ocean_proximity",axis=1)


# In[59]:


imputer.fit(housing_num)


# In[60]:


housing_num.describe()


# In[61]:


imputer.statistics_


# In[62]:


X = imputer.transform(housing_num)


# In[39]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# In[40]:


housing_tr.head(5)


# In[52]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded


# In[54]:


encoder.classes_


# In[55]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1)) 
housing_cat_1hot


# In[56]:


housing_cat_1hot.toarray()


# In[57]:


from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat) 
housing_cat_1hot


# In[58]:


housing_cat.head(5)


# In[80]:


housing.columns


# In[59]:


from sklearn.base import BaseEstimator, TransformerMixin

# get the right column indices: safer than hard-coding indices 3, 4, 5, 6
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[85]:


from sklearn.preprocessing import FunctionTransformer

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)


# In[87]:


housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()


# In[88]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[92]:


housing_num_tr


# In[93]:


from sklearn.compose import ColumnTransformer


# In[94]:


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)


# In[95]:


housing_prepared


# In[96]:


housing_prepared.shape


# # #TRAINING

# In[99]:


from sklearn.linear_model import LinearRegression 

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[123]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))


# In[124]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions) 
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[125]:


from sklearn.tree import DecisionTreeRegressor 
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[126]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions) 
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[127]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10) 
tree_rmse_scores = np.sqrt(-scores)


# In[132]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean  :", scores.mean())
    print("STD   :", scores.std())


# In[133]:


display_scores(tree_rmse_scores)


# In[134]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[136]:


from sklearn.ensemble import RandomForestRegressor  
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)


# In[137]:


housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions) 
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[138]:


display_scores(forest_rmse)


# In[142]:


from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, 
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)


# In[143]:


grid_search.best_params_


# In[147]:


grid_search.best_estimator_
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                      max_features=8, max_leaf_nodes=None, min_impurity_split=1e-07,
                      min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=30, n_jobs=1, oob_score=False, random_state=42, verbose=0,
                      warm_start=False)


# In[148]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  
    print(np.sqrt(-mean_score), params)


# In[ ]:




