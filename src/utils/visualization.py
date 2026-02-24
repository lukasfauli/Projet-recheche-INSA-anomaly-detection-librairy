'''In this file, you will find all the the functions that will help you to visualize your data'''

from  matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def plot(df, col_name_X, col_name_Y, ax=None):
     '''
     Plotting function with optional axis to support subplots
     ''' 

     created_fig = False
     if ax is None:
          fig, ax = plt.subplots()
          created_fig = True

     ax.plot(df[col_name_X], df[col_name_Y])
     ax.set_xlabel(f"Values of {col_name_X}")
     ax.set_ylabel(f"Values of {col_name_Y}")
     ax.set_title(f"{col_name_Y} in function of {col_name_X}")

     if created_fig:
          plt.show()
    
def plot_hist(df, col_name_X, bins=100, ax=None):
     '''
     Representation in the form of a histogram to see possible discontinuity in data
     Accepts an optional `ax` to draw into an existing subplot.
     '''
     created_fig = False
     if ax is None:
          fig, ax = plt.subplots()
          created_fig = True

     ax.hist(df[col_name_X], bins=bins, color='skyblue', edgecolor='black')
     ax.set_yscale("log")   # use log scale to flatten peaks and raise small values
     ax.set_ylabel("Number of occurence (Log)")
     ax.set_title(f"Discontinuty analysis of {col_name_X}")

     if created_fig:
          plt.show()

def boxplot(df,col_name,message=True, rotation=0):
     sns.boxplot(data=df[col_name])
     if message:
          plt.title(f"Boxplot Analysis of {col_name}")
          plt.xlabel(f"Values of {col_name}")
     plt.xticks(rotation=rotation)
     plt.show()