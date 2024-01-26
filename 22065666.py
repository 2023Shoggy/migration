# -*- coding: utf-8 -*-
# !pip install lmfit
# !pip install cluster_tools==1.61
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.cluster as cluster
import sklearn.metrics as skmet
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from scipy.stats import t
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

from lmfit import Model
import cluster_tools as ct
import scipy.optimize as opt
import itertools as iter


def read_data(filename, skiprows_start=4, skiprows_end=5, **others):
    """
    A function that reads climate change data and returns the dataset with the first 4 rows skipped
    and the specified number of rows skipped from the end.

    Parameters:
        filename (str): The name of the world bank data file.
        skiprows_start (int): The number of rows to skip at the beginning.
        skiprows_end (int): The number of rows to skip at the end.
        **others: Additional arguments to pass into the function.

    Returns:
        pd.DataFrame: The dataset with the specified number of rows skipped.
    """
    world_data = pd.read_csv(filename, **others)

    # Determine the number of rows to keep from the end
    rows_to_keep = world_data.shape[0] - skiprows_end

    # Keep only the specified number of rows from the end
    world_data = world_data.iloc[:rows_to_keep]

    return world_data

# Read data with 4 rows skipped at the beginning and 5 rows skipped at the end
net_mig = read_data('./f367f820-25a7-4e06-9eb7-9c9072fd870c_Data.csv', skiprows_start=4, skiprows_end=5)

# Now, Net Migration contains the dataset with the first 4 rows skipped and 5 rows skipped from the end

net_mig.head()

net_mig.describe

# Dropping the columns not needed
net_mig = net_mig.drop(['Country Code', 'Series Code'], axis=1)

# reseting the index
net_mig.reset_index(drop=True, inplace=True)

# Selecting  the countries the needed
net_mig = net_mig[net_mig['Country Name'].isin(['Nigeria', 'United Kingdom', 'India', 'Brazil', 'China', 'United States',
                                                'Australia', 'Kenya', 'Ghana', 'Saudi Arabia', 'Argentina', 'Israel'])]

net_mig

# Assuming net_mig is your DataFrame
net_mig = net_mig.drop('2022 [YR2022]', axis=1)  # Drop the '2022 [YR2022]' column

# Display the resulting DataFrame
print(net_mig)

net_mig.head(12).corr

# Assuming net_mig is your DataFrame
years = ['1990 [YR1990]', '2000 [YR2000]', '2010 [YR2010]', '2020 [YR2020]']

# Extracting the relevant columns
net_migs = net_mig[['Country Name'] + years]

# Plotting
plt.figure(figsize=(12, 8))

for index, row in net_migs.head(12).iterrows():
    plt.plot(years, row[years], label=row['Country Name'])

plt.title('Net Migration Trends (1990, 2000, 2010, 2020)')
# Update x-axis ticks
plt.xticks(years, ['1990', '2000', '2010', '2020'])
plt.xlabel('Year')
plt.ylabel('Net Migration')
plt.legend(loc='upper right')  # Move the legend to the top right corner
plt.show()

# Assuming net_mig is your DataFrame
years = ['1990 [YR1990]', '2000 [YR2000]', '2010 [YR2010]', '2020 [YR2020]']

# Extracting the relevant columns
net_migs = net_mig[['Country Name'] + years]

# Plotting a bar chart
plt.figure(figsize=(12, 8))

# Set up positions for the bars
bar_width = 0.2
bar_positions = np.arange(len(net_migs))

# Plot bars for each year
for i, year in enumerate(years):
    plt.bar(bar_positions + i * bar_width, net_migs[year], width=bar_width, label=f'{year[-5:-1]}')

plt.xlabel('Country')
plt.ylabel('Net Migration')
plt.title('Net Migration Trends (1990, 2000, 2010, 2020) by Country')
plt.xticks(bar_positions + (len(years) - 1) * bar_width / 2, net_migs['Country Name'], rotation=45, ha='right')
plt.legend()
plt.show()

import pandas as pd
from scipy.optimize import curve_fit
import numpy as np

# Load the CSV data
data_path = "f367f820-25a7-4e06-9eb7-9c9072fd870c_Data.csv"
data = pd.read_csv(data_path, header=0)

# Replace ".." with  and ensure to drop  values before conversion
# data = data.replace('..', np.)
data = data.replace('..', np.nan)

# Filter data for the United Kingdom, ensuring to drop  values before conversion
uk_data = data[data['Country Code'] == 'DEU'].iloc[0, 5:].dropna().astype(float)

# Correctly extract the year from the column names by removing non-numeric characters
years = np.array([int(year.replace(' [YR', '').replace(']', '')) for year in uk_data.index])

# Net migration data
net_migration = uk_data.values

# Define a polynomial function to fit
def poly_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Fit the model
popt, pcov = curve_fit(poly_func, years, net_migration)

# Generate predictions for the next 10 and 20 years with the fitted model
future_years = np.array([2023 + i for i in range(1, 21)])
predictions = poly_func(future_years, *popt)

# Calculate confidence intervals for the predictions
perr = np.sqrt(np.diag(pcov))
upper_bound = poly_func(future_years, *(popt + 1.96 * perr))
lower_bound = poly_func(future_years, *(popt - 1.96 * perr))

print('Model coefficients:', popt)
print('Predictions for the next 20 years:', predictions)
print('Confidence intervals:', np.vstack([lower_bound, upper_bound]))

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(years, net_migration, color='blue', label='Actual Net Migration')
plt.plot(future_years, predictions, color='red', label='Predicted Net Migration')
plt.fill_between(future_years, lower_bound, upper_bound, color='grey', alpha=0.2, label='Confidence Interval')
plt.title('Predicted Net Migration for the United Kingdom (2023-2042)')
plt.xlabel('Year')
plt.ylabel('Net Migration')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

# Assuming net_mig is your DataFrame
years = ['1990 [YR1990]', '2000 [YR2000]', '2010 [YR2010]', '2020 [YR2020]']

# Extracting the relevant columns
net_yr = net_mig[['Country Name'] + years]

# Checking the summary statistics for the selected years
net_yr_description = net_yr[years].describe()
print(net_yr_description)

# # Normalizing the data and storing minimum and maximum value

# Assuming net_mig is your DataFrame
years = ['1990 [YR1990]', '2020 [YR2020]']

# Extracting the relevant columns
net_yr2 = net_mig[['Country Name'] + years]

# Print the extracted DataFrame
print(net_yr2)

# Assuming net_yr2 is your DataFrame
scaler = StandardScaler()
net_yr2_norm = scaler.fit_transform(net_yr2[years])

# Get the minimum and maximum values
net_yr2_min = net_yr2_norm.min()
net_yr2_max = net_yr2_norm.max()

# Print normalized data statistics
print(pd.DataFrame(net_yr2_norm, columns=years).describe())

# Extracting 30 years of data at an interval of 10 years from the dataset
net_yr2 = net_migs[['1990 [YR1990]', '2000 [YR2000]', '2010 [YR2010]', '2020 [YR2020]']]
net_yr2.describe()

# Checking for any missing values
net_yr2.isna().sum()

# Checking for correlation between our years choosen

# Correlation
corr = net_yr2.corr()
corr

# Assuming net_mig is your DataFrame
years = ['1990 [YR1990]', '2000 [YR2000]', '2010 [YR2010]', '2020 [YR2020]']

# Extracting the relevant columns
net_yr = net_mig[['Country Name'] + years]

# Drop the 'Country Name' column for clustering
X = net_yr[years]

# Standardize the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculating the best clustering number using silhouette score
for i in range(2, 9):
    # Creating kmeans and fit
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)

    # Extract labels and calculate silhouette score
    labels = kmeans.labels_
    score = silhouette_score(X_scaled, labels)

    print(f'Number of clusters: {i}, Silhouette Score: {score}')

# Extract the columns for clustering
ext_clus = net_yr2[['2010 [YR2010]', '2020 [YR2020]']]
net_yr2_norm = scaler.fit_transform(ext_clus)  # Use scaler on the selected columns

# Plotting the Clusters
nclusters = 3  # number of cluster centres

kmeans = KMeans(n_clusters=nclusters, random_state=42)
kmeans.fit(net_yr2_norm)

# Extract labels and cluster centers
labels = kmeans.labels_
cen = kmeans.cluster_centers_

# Scatter plot with colors selected using the cluster numbers
plt.figure(figsize=(10, 6))

plt.scatter(ext_clus["2010 [YR2010]"], ext_clus["2020 [YR2020]"], c=labels, cmap="tab10")

# Show cluster centres
xcen = cen[:, 0]
ycen = cen[:, 1]

plt.scatter(xcen, ycen, c="k", marker="d", s=80)
# c = color, s = size

plt.xlabel("Net Migration (1990)")
plt.ylabel("Net Migration (2020)")
plt.title("Clusters for Net Migration")
plt.show()


# Reshaping the dataset to have 'Year' and 'Value' columns
years = [str(year) for year in range(1990, 2021)]
year_columns = [f'{year} [YR{year}]' for year in years]

# Melting the dataframe to have a long format
data_long = pd.melt(data, id_vars=['Series Name', 'Series Code', 'Country Name', 'Country Code'], value_vars=year_columns, var_name='Year', value_name='Value')

# Extracting year from 'Year' column
data_long['Year'] = data_long['Year'].apply(lambda x: x.split(' ')[0])

# Imputing missing values
imputer = SimpleImputer(strategy='mean')
data_long['Value'] = imputer.fit_transform(data_long[['Value']])

# Standardizing the 'Value' feature
scaler = StandardScaler()
data_long['Value_scaled'] = scaler.fit_transform(data_long[['Value']])

# Applying KMeans clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(data_long[['Year', 'Value_scaled']])
data_long['Cluster'] = kmeans.predict(data_long[['Year', 'Value_scaled']])

# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(data_long['Year'], data_long['Value'], c=data_long['Cluster'], cmap='viridis')
plt.title('Scatter Plot of Data by Year with Cluster Coloring')
plt.xlabel('Year')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()
print('Scatter plot generated with different colors representing clusters between years.')

def plot_predicted(country_name):
    # Load the dataset
    file_path = 'f367f820-25a7-4e06-9eb7-9c9072fd870c_Data.csv'
    data = pd.read_csv(file_path, encoding='ascii')

    # Filter data for United Kingdom
    uk_data = data[data['Country Name'] == country_name].iloc[0, 5:].replace('..', np.nan).astype(float)

    # Prepare data from 1990 onwards
    years = np.array([int(year.split(' ')[0]) for year in data.columns[5:] if int(year.split(' ')[0]) >= 1990])
    values = uk_data.values[len(uk_data.values)-len(years):]

    # Remove NaN values from 'years' and 'values'
    valid_indices = ~np.isnan(values)
    years = years[valid_indices]
    values = values[valid_indices]

    # # Polynomial regression
    degree = 3
    polyreg_cleaned = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg_cleaned.fit(years.reshape(-1, 1), values)

    # # Predict future values
    future_years = np.arange(years[-1] + 1, years[-1] + 21).reshape(-1, 1)
    # future_values = polyreg.predict(future_years)

    # # Calculate confidence interval
    # y_pred = polyreg.predict(years.reshape(-1, 1))
    # se = np.sqrt(mean_squared_error(values, y_pred))
    t_val = t.ppf(0.975, len(years)-degree-1)
    # confidence_interval = t_val * se * np.sqrt(1/len(years) + (future_years - np.mean(years))**2 / np.sum((years - np.mean(years))**2))



    # Check for NaN values in the dataset and handle them
    values_cleaned = np.nan_to_num(values, nan=np.nanmean(values))

    # Predict future values with cleaned data
    future_values_cleaned = polyreg_cleaned.predict(future_years)

    # Calculate confidence interval for cleaned data
    y_pred_cleaned = polyreg_cleaned.predict(years.reshape(-1, 1))
    se_cleaned = np.sqrt(mean_squared_error(values_cleaned, y_pred_cleaned))
    confidence_interval_cleaned = t_val * se_cleaned * np.sqrt(1/len(years) + (future_years - np.mean(years))**2 / np.sum((years - np.mean(years))**2))

    # Correcting the confidence interval calculation to match the shape of future_values_cleaned
    confidence_interval_cleaned = confidence_interval_cleaned.reshape(-1)

    # Plotting with cleaned data
    plt.figure(figsize=(10, 6))
    plt.scatter(years, values_cleaned, color='blue', label='Actual Data (Cleaned)')
    plt.plot(years, y_pred_cleaned, color='green', label='Regression Line (Cleaned)')
    plt.plot(future_years, future_values_cleaned, 'r--', label='Predicted Data (Cleaned)')
    plt.fill_between(future_years.ravel(), (future_values_cleaned - confidence_interval_cleaned), (future_values_cleaned + confidence_interval_cleaned), color='red', alpha=0.2, label='95% Confidence Interval (Cleaned)')
    plt.xlabel('Year')
    plt.ylabel('Net Migration')
    plt.title(f'{country_name} Net Migration Prediction with Polynomial Regression (Cleaned)')
    plt.legend()

plot_predicted('United Kingdom')
plot_predicted('India')