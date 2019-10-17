import pandas as pd
import numpy as np
from scipy.stats import zscore

election = pd.read_csv('/Users/licheng/Training/data/elections.csv', index_col='county')

#*************** SLICE DATA FRAMES *********************************************
# Slice the row labels 'Potter' to 'Perry' in reverse order: p_counties_rev
p_counties_rev = election.loc['Potter':'Perry':-1,:]
print(p_counties_rev)

# Slice the columns from the starting column to 'Obama': left_columns
left_columns = election.loc[:,:'Obama']
print(left_columns.head())

# Slice the columns from 'Obama' to 'winner': middle_columns
middle_columns = election.loc[:,'Obama':'winner']
print(middle_columns.head())

# Slice the columns from 'Romney' to the end: 'right_columns'
right_columns = election.loc[:,'Romney':]
print(right_columns.head())

# Create the list of row labels: rows
rows = ['Philadelphia', 'Centre', 'Fulton']

# Create the list of column labels: cols
cols = ['winner', 'Obama', 'Romney']

# Create the new DataFrame: three_counties
three_counties = election.loc[rows,cols]
print(three_counties)

#*************** FILTERING DATA FRAMES *********************************************

high_turnout_df = election[election.turnout>70]
print(high_turnout_df)

# Create the boolean array: too_close
too_close = election.margin < 1

# Assign np.nan to the 'winner' column where the results were too close to call
election.winner[election.margin < 1] = np.nan
print(election.info())

titanic = pd.read_csv('/Users/licheng/Training/data/titanic.csv')

# Select the 'age' and 'cabin' columns: df
df = titanic[['Age','Cabin']]

# Print the shape of df
print(df.shape)

# Drop rows in df with how='any' and print the shape
print(df.dropna(how='any').shape)

# Drop rows in df with how='all' and print the shape
print(df.dropna(how='all').shape)

# Drop columns in titanic with less than 1000 non-missing values
print(titanic.dropna(thresh=1000, axis='columns').info())

weather = pd.read_csv('/Users/licheng/Training/data/weather.csv')

#******************* TRANSFORM DATA FRAMES ***************************************
# Write a function to convert degrees Fahrenheit to degrees Celsius: to_celsius
def to_celsius(F):
    return 5/9*(F - 32)

# Apply the function over 'Mean TemperatureF' and 'Mean Dew PointF': df_celsius
df_celsius = weather[['Mean TemperatureF','Mean Dew PointF']].apply(to_celsius)
# Reassign the column labels of df_celsius
df_celsius.columns = ['Mean TemperatureC', 'Mean Dew PointC']

print(df_celsius.head())

#Using .map() with a dictionary
# Create the dictionary: red_vs_blue
red_vs_blue = {'Obama':'blue','Romney':'red'}

# Use the dictionary to map the 'winner' column to the new column: election['color']
election['color'] = election.winner.map(red_vs_blue)

print(election.head())

# Using vectorized functions
# Call zscore with election['turnout'] as input: turnout_zscore
turnout_zscore = zscore(election['turnout'])
print(type(turnout_zscore))

# Assign turnout_zscore to a new column: election['turnout_zscore']
election['turnout_zscore'] = turnout_zscore
print(election.head())

sales = pd.read_csv('/Users/licheng/Training/data/sales.csv', index_col='month')

# Changing index of a DataFrame
# Create the list of new indexes: new_idx
new_idx = [sales.index[i].upper() for i in range(len(sales.index))]

# Assign new_idx to sales.index
sales.index = new_idx
print(sales)

# Changing index name labels
# Assign the string 'MONTHS' to sales.index.name
sales.index.name = 'MONTHS'
print(sales)

# Assign the string 'PRODUCTS' to sales.columns.name
sales.columns.name = 'PRODUCTS'
print(sales)

# Building an index, then a DataFrame
# Generate the list of months: months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

# Assign months to sales.index
sales.index = months
print(sales)

users = pd.read_csv('/Users/licheng/Training/data/users.csv')

# Pivoting a single variable
# Pivot the users DataFrame: visitors_pivot
visitors_pivot = users.pivot(index = 'weekday',columns='city',values='visitors')
print(visitors_pivot)

# Pivot users with signups indexed by weekday and city: signups_pivot
signups_pivot = users.pivot(index = 'weekday',columns = 'city',values = 'signups')
print(signups_pivot)

# Pivot users pivoted by both signups and visitors: pivot
pivot = users.pivot(index = 'weekday',columns = 'city')
print(pivot)


