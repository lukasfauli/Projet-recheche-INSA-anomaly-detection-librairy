'''In this file, you will find all the the functions that will help you to visualize your data'''

from  matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def plot(df, col_name_X,col_name_Y):
     '''
     Plotting function 
     ''' 

     plt.plot(df[col_name_X],df[col_name_Y])
     plt.xlabel(f"Values of {col_name_X}")
     plt.ylabel(f"Values of {col_name_Y}")
     plt.title( f"{col_name_X} in function of {col_name_Y}")
     plt.show()
    
def plot_hist(df,col_name_X,bins=100):
     '''
     Representation in the form of a histogram to see possible discontinuity in data
     '''
     plt.hist(df[col_name_X], bins=bins, color='skyblue', edgecolor='black')
     plt.yscale("log")   #the use of the log scale is to flat the peaks and raise the small values.
     plt.ylabel("Number of occurence (Log)")
     plt.title(f"Discontinuty analysis of {col_name_X}")
     plt.show()

def boxplot(df,col_name):
     sns.boxplot(x=df[col_name])
     plt.title(f"Boxplot Analysis of {col_name}")
     plt.xlabel(f"Values of {col_name}")
     plt.show()