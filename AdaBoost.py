import pandas as pd
import sqlite3 as sql
import numpy as np
import matplotlib.pyplot as plt
#from scipy.fftpack import cs_diff, sc_diff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
#import streamlit as st

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import matplotlib.image as mpimg

#import access_name
import plotly.express as px
#import code.streamlit_app as st
#import access_name
import requests
import urllib
#import cv2
import os
from sklearn.model_selection import RepeatedKFold

import subprocess
import sys
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
from numpy import mean
from numpy import std

csv_path = ('MAIN.csv')
df = pd.read_csv(csv_path)


df = df[['batter', 'pitch_type', 'p_throws', 'zone', 'release_spin_rate', 'balls', 'strikes',
         'release_speed', 'estimated_ba_using_speedangle']]

#'SELECT * from EVENT WHERE player_name=Junis, Jakob'
player_serial = 608369

df = pd.get_dummies(df)
y = df['estimated_ba_using_speedangle']
df = df[df['batter'] == player_serial]
df = df.drop('batter', axis=1)

columns = df.shape[1]
# st.write(sim_dict)


# 3st.dataframe(df)
#########################################
#########################################
# print(sql_query)
df = df[df['balls'].notna()]
df = df[df['strikes'].notna()]

#df = df.append(simulated_pitch)
#df = pd.get_dummies(df)

#df = df[df['launch_speed'].notna()]
df = df[df['estimated_ba_using_speedangle'].notna()]
df = df[df['release_spin_rate'].notna()]

row_count = df.count
#sim_pitch = df.iloc[-1]
#df = df[:-1]
#X = df.drop('player_name')
# st.write(df)
y = df['estimated_ba_using_speedangle']

# st.write(df)
X = df.drop('estimated_ba_using_speedangle', axis=1)


X, y = make_regression(n_samples=row_count, n_features=columns,
                       n_informative=8, noise=0.1, random_state=6)

model = AdaBoostRegressor()


cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(
    model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
