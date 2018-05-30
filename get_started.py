import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

print(os.listdir("data/"))

app_train = pd.read_csv('data/application_train.csv')
print('Training data shape: ', app_train.shape)
app_train.head()

#train has one more column target, which indicate the pay or no pay status.

print(app_train['TARGET'].value_counts())
#Training data shape:  (307511, 122)
# 0    282686
# 1     24825
# of course many more people pays their debts.

# Missing values statistics
missing_values = missing_values_table(app_train)
missing_values.head(20)
print(missing_values)
# preprocess of data is super important in machine learning. Garbage in garbage out.
# Task 1 prepocessing missing value

