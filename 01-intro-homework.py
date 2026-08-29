# %% [markdown]
# # ML Zoomcamp 2025
# ## Module 1 - Introduction Homework

# %% [markdown]
# ## Q1. Pandas version
# 
# What version of Pandas did I install?

# %%
import pandas as pd
import numpy as np

pd.__version__

# %% [markdown]
# ## Loading the dataset

# %%
df = pd.read_csv('car_fuel_efficiency.csv')
df.head()

# %% [markdown]
# ## Q2. Records count
# 
# How many records are in the dataset?

# %%
df.shape


# %% [markdown]
# ## Q3. Fuel types

# %%
df['fuel_type'].nunique()

# %% [markdown]
# ## Q4. Missing values

# %%
df.isnull().sum()
#4 

# %% [markdown]
# ## Q5. Max fuel efficiency

# %%
df[df['origin']=='Asia']['fuel_efficiency_mpg'].max()
#23.75

# %% [markdown]
# ## Q6. Median value of horsepower

# %% [markdown]
# ## 6-1. the median value of the horsepower

# %%
df['horsepower'].mean()  #149.65

# %% [markdown]
# ## 6.2-the most frequent value of the horsepower

# %%
df['horsepower'].value_counts()    #152

# %% [markdown]
# ## 6.3-fill the missing values with the most frequent value.

# %%
df['horsepower'].fillna(152)


# %% [markdown]
# ## 6.4-the median value of horsepower once again.

# %%
df.horsepower.mean()
# it hasn't changed because we didn't assign the filled values back to the dataframe. We need to do df['horsepower'] = df['horsepower'].fillna(152) to update the dataframe.

# %% [markdown]
# ## Q7. Sum of weights

# %% [markdown]
# ## 7.1

# %%
asia_cars = df[df['origin']=='Asia']
asia_cars

# %% [markdown]
# ## 7.2

# %%
asia_cars = asia_cars.loc[:,['vehicle_weight' , 'model_year']]
asia_cars

# %% [markdown]
# ## 7.3

# %%
asia_cars=asia_cars.head(7)
asia_cars

# %% [markdown]
# ## 7.4

# %%
X = asia_cars.to_numpy()
X

# %% [markdown]
# ## 7.5

# %%
XTX= X.T.dot(X)
XTX

# %% [markdown]
# ## 7.6

# %%
XTX.shape  # it should be symmetric and square, so it should be (2,2) because we have 2 features in the dataset.

# %%
inv = np.linalg.inv(XTX)
inv

# %% [markdown]
# ## 7.7

# %%
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
y

# %% [markdown]
# ## 7.8

# %%
w=inv.dot(X.T).dot(y)
w

# %% [markdown]
# ## 7.9

# %%
np.sum(w)  #0.51


