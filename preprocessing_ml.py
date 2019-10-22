import pandas as pd 
import numba as np
import os
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import re

volunteer = pd.read_csv('/Users/licheng/Training/data/volunteer.csv')
volunteer.head()

# Missing value treatment ********************************************

# Check how many values are missing in the category_desc column
volunteer['category_desc'].isnull().sum()

# Subset the volunteer dataset
volunteer_subset = volunteer[volunteer['category_desc'].notnull()]

# Print out the shape of the subset
volunteer_subset.shape

# Data types and conversions *******************************************
volunteer.dtypes

# Convert the hits column to type int 
volunteer_2 = volunteer["hits"].astype('object')
volunteer_2.dtypes

# Class distribution *********************************************************
# Category counts
volunteer["category_desc"].value_counts()

# Create a data with all columns except category_desc
volunteer_X = volunteer.drop('category_desc', axis=1)

# Create a category_desc labels dataset
volunteer_y = volunteer[['category_desc']]

# Use stratified sampling to split up the dataset according to the volunteer_y dataset
X_train, X_test, y_train, y_test = train_test_split(volunteer_X, volunteer_y, stratify = volunteer_y)

# Stantarization *********************************************************
wine = pd.read_csv('/Users/licheng/Training/data/wine.csv')

# The variance of the column Proline is extremely high
wine.var()

# Apply the log normalization function to the Proline column
wine['Proline_log'] = np.log(wine['Proline'])

# Scaling data (convert to Nor(0,1) *********************************************************

# Quick general statistic description
wine.describe()

# Create the scaler
ss = StandardScaler()

# Take a subset of the DataFrame you want to scale 
wine_subset = wine[['Ash','Acl','Mg']]

# Apply the scaler to the DataFrame subset
wine_subset_scaled = pd.DataFrame(ss.fit_transform(wine_subset), columns=wine_subset.columns)
wine_subset_scaled.describe()

# Before modeling, scaling *********************************************************

X = wine.drop('Wine',axis = 1)
y = wine['Wine']

# Create the scaling method.
ss = StandardScaler()

# Apply the scaling method to the dataset used for modeling.
X_scaled = ss.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# Fit the k-nearest neighbors model to the training data.
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

# Score the model on the test data.
print(knn.score(X_test,y_test))

# Encoding categorical variables *********************************************************
hiking = pd.read_json('/Users/licheng/Training/data/hiking.json')

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
hiking['Accessible_enc'] = enc.fit_transform(hiking['Accessible'])

# Compare the two columns
print(hiking[['Accessible', 'Accessible_enc']].head())

# Encoding categorical variables - one-hot
# Transform the category_desc column
category_enc = pd.get_dummies(volunteer["category_desc"])

# Take a look at the encoded columns
print(category_enc.head())

# Engineering numerical features - taking an average
running_times_5k = pd.DataFrame(
                    {'name':['Sue','Mark','Sean','Erin','Jenny','Russell'],
                     'run1':[20.1,16.5,23.5,21.7,25.8,30.9],
                     'run2':[18.5,17.1,25.1,21.1,27.1,29.6],
                     'run3':[19.6,16.9,25.2,20.9,26.1,31.4],
                     'run4':[20.3,17.6,24.6,22.1,26.7,30.4],
                     'run5':[18.3,17.3,23.9,22.2,26.9,29.9]
                    })

# Create a list of the columns to average
run_columns = running_times_5k.columns[1:6]

# Use apply to create a mean column
running_times_5k["mean"] = running_times_5k.apply(lambda row: row[run_columns].mean(), axis=1)

# Take a look at the results
print(running_times_5k)

# Engineering numerical features - datetime
# First, convert string column to date column
volunteer["start_date_converted"] = pd.to_datetime(volunteer["start_date_date"])

# Extract just the month from the converted column
volunteer["start_date_month"] = volunteer["start_date_converted"].apply(lambda row: row.month)

# Take a look at the converted and new month columns
print(volunteer[["start_date_converted","start_date_month"]].head())

# Engineering features from strings - extraction
# Write a pattern to extract numbers and decimals
def return_mileage(length):
    pattern = re.compile(r"\d+\.\d+")
    # \d+ -> every digit possible
    # \. dot 

    # Search the text for matches
    mile = re.match(length,pattern)
    
    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        return float(mile.group(0))
        
# Apply the function to the Length column and take a look at both columns
hiking["Length_num"] = hiking["Length"].apply(lambda row: return_mileage(row))
print(hiking[["Length", "Length_num"]].head())




strin = "temperature: 75.6 F"
pat  = re.compile(r"(\d+) (\.) (\d+)")
temp = re.match(pat,strin)
temp.group(0)

m = re.match(r"(\w+) (\w+)", "Isaac Newton, physicist")
m.group(0)


# Putting it all together ===============================================================
ufo = pd.read_csv('/Users/licheng/Training/data/ufo.csv')

# Check the column types
print(ufo.dtypes)

# Change the type of seconds to float
ufo["seconds"] = ufo["seconds"].astype(float)

# Change the date column to type datetime
ufo["date"] = pd.to_datetime(ufo["date"])

# Check the column types
print(ufo[["seconds", "date"]].dtypes)

# Check how many values are missing in the length_of_time, state, and type columns
print(ufo[['length_of_time', 'state', 'type']].isnull().sum())

# Keep only rows where length_of_time, state, and type are not null
ufo = ufo[ufo['length_of_time'].notnull() &
          ufo['state'].notnull() &
          ufo['type'].notnull()]

# Print out the shape of the new dataset
print(ufo.shape)

# Extracting numbers from strings
def return_minutes(time_string):
    # Use \d+ to grab digits
    pattern = re.compile(r"\d+")

    # Use match on the pattern and column
    num = re.match(pattern, time_string)
    if num is not None:
        return int(num.group(0))


# Apply the extraction to the length_of_time column
ufo["minutes"] = ufo["length_of_time"].apply(lambda row: return_minutes(row))

# Take a look at the head of both of the columns
print(ufo[['length_of_time', 'minutes']].head())

# Identifying features for standardization
# Check the variance of the seconds and minutes columns
ufo = ufo[ufo['seconds']!=0] # This line is not on the training

print(ufo[['seconds','minutes']].var())

# Log normalize the seconds column
ufo["seconds_log"] = np.log(ufo['seconds'])

# Print out the variance of just the seconds_log column
print(ufo['seconds_log'].var())

# Encoding categorical variables
# Use Pandas to encode us values as 1 and others as 0
ufo["country_enc"] = ufo["country"].apply(lambda row: 1 if row == 'us' else 0)

# Print the number of unique type values
print(len(ufo.type.unique()))

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo.type)

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([ufo, type_set], axis=1)

# Features from dates
# Look at the first 5 rows of the date column
print(ufo["date"].head())

# Extract the month from the date column
ufo["month"] = ufo["date"].dt.month

# Extract the year from the date column
ufo["year"] = ufo["date"].dt.year

# Take a look at the head of all three columns
print((ufo[['date', 'month', 'year']]))

# Text vectorization
# Take a look at the head of the desc field
print(ufo['desc'].head())

# Create the tfidf vectorizer object
vec = TfidfVectorizer()

# Use vec's fit_transform method on the desc field
desc_tfidf = vec.fit_transform(ufo.desc)

# Look at the number of columns this creates
print(desc_tfidf.shape)

# Selecting the ideal dataset
# Check the correlation between the seconds, seconds_log, and minutes columns
print(ufo[['seconds','seconds_log','minutes']].corr())

# Make a list of features to drop
to_drop = ['city', 'country', 'lat','long', 'state', 'seconds', 'minutes', 'date', 'desc', 'length_of_time', 'recorded']

# Drop those features
ufo_dropped = ufo.drop(to_drop, axis=1)

# Let's also filter some words out of the text vector we created
filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)



