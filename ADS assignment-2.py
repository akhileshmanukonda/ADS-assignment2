# TASK 1: Ingest and manipulate the data using pandas dataframes

import pandas as pd
import matplotlib.pyplot as plt0
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt
import seaborn as sns


# This example uses the World Bank data on renewable energy consumption as a
# percentage of total final energy consumption for all countries:
def read_worldbank_data(filename):
    """
    Reads data from a CSV file containing World Bank data on renewable energy
    consumption as a percentage of total final energy consumption
    for all countries.

    Parameters:
    filename (str): The name of the CSV file containing the data.

    Returns:
    tuple: A tuple containing two pandas DataFrames. The first DataFrame is a
    year-based DataFrame with each column representing a country and each row
    representing a year. The second DataFrame is a country-based DataFrame with
    each row representing a country and each column representing a year.
    """
    df = pd.read_csv(filename, skiprows=4)

    df_country = df.drop(columns=['Country Code', 'Indicator Code',
                                  'Unnamed: 66'], inplace=True)
    df_country = df.set_index('Country Name').T
    df_year = df.set_index('Country Name').reset_index()

    return df_year, df_country


df_year, df_country = read_worldbank_data('WorldBank.csv')
df_year.head()
# Getting  first 5 outputs
df_country
# Getting output

#decribes the data frame of year
a = df_year.describe()
#printing the values of describe are count,mean,std,min,max
print(a)
#decribes the data frame of country
s = df_country.describe()
#printing the values of describe are count,mean,std,min,max
print(s)

def read_worldbank_data(filename):
    """
        Reads data from a CSV file containing World Bank data on various
        indicators for all countries, including urban population.

        Parameters:
        filename (str): The name of the CSV file containing the data.

        Returns:
        tuple: A tuple containing two pandas DataFrames. The first DataFrame
        is a year-based DataFrame with each column representing indicator and
        each row representing a year. The second DataFrame is a indicator-based
        DataFrame with each row representing an indicator and each column
        representing a country.
    """
    df2 = pd.read_csv(filename, skiprows=4)

    df_country2 = df2.drop(columns=['Country Code', 'Indicator Code',
                                    'Unnamed: 66'], inplace=True)
    df_year2 = df2.set_index('Country Name').T

    df_country2 = df2.set_index('Indicator Name').reset_index()

    return df_year2, df_country2


# Read the World Bank data from a CSV file
df_year2, df_country2 = read_worldbank_data('WorldBank.csv')
df_year2
# Getting output


def attribute_function(indicators, df):
    """
        Filter the rows of a DataFrame based on whether the 'Indicator Name'
        column contains the given value.

        Parameters:
        indicators (str): The value to search for in the 'Indicator Name'
        column.
        df (pandas.DataFrame): The DataFrame to filter.

        Returns:
        pandas.DataFrame: A new DataFrame containing only the rows where the
        'Indicator Name' column contains the given value.
    """

    df = df[df['Indicator Name'].isin([indicators])]
    return df


def country(cty, df):
    """
        Filter the rows of a DataFrame based on whether the 'Country Name'
        column contains the given value, set the 'Indicator Name' column as the
        index, drop the 'Country Name' column, and transpose the resulting
        DataFrame.

        Parameters:
        cty (str): The value to search for in the 'Country Name' column.
        df (pandas.DataFrame): The DataFrame to filter.

        Returns:
        pandas.DataFrame: A new DataFrame containing only the rows where the
        'Country Name' column contains the given value, with the
        'Indicator Name' column set as the index, the 'Country Name' column
        dropped, and the resulting DataFrame transposed.
    """
    df = df[df['Country Name'].isin([cty])]
    df = df.set_index("Indicator Name")
    df = df.drop("Country Name", axis=1)
    df = df.T
    return df


# Create a new DataFrame called US_pop_Compare by filtering df_country2 for
# rows where the 'Country Name' column contains 'United States'
US_pop_Compare = country("United States", df_country2)

# Select a subset of columns from US_pop_Compare and assign it to a new
# DataFrame called total_pop
total_pop = US_pop_Compare[["Population in urban agglomerations of more " +
                            "than 1 million (% of total population)",
                           "Population living in areas where elevation is " +
                            "below 5 meters (% of total population)",
                            "Urban population living in areas " +
                            "where elevation is below 5 meters " +
                            "(% of total population)",
                            "Rural population living in areas where " +
                            "elevation is below 5 meters " +
                            "(% of total population)"]]

# Compute the pairwise correlation of columns in total_pop, excluding NA/null
#values
ed = total_pop.corr()

# Return the resulting correlation matrix
ed


# Create a heatmap of the correlation matrix with annotations showing the
#correlation coefficients
ax = sns.heatmap(ed, annot=True)

# Call the attribute_function with the arguments "Population in urban
#agglomerations of more than 1 million (% of total population)" and df_country2
urban_agglo = attribute_function("Population in urban agglomerations of " +
                                 "more than 1 million (% of total population)",
                                 df_country2)

# Assign the resulting DataFrame to the variable urban_agglo


# Create a list of countries to plot
countries = ['Australia', 'United States', 'United Kingdom', 'World']

# Filter the urban_agglo DataFrame to only include rows where the
#"Country Name" column is in the countries list
urban_agglo = urban_agglo[urban_agglo['Country Name'].isin(countries)]

# Create a step plot of the data
plt0.step(urban_agglo['Country Name'], urban_agglo['2015'])

# Rotate the x-axis labels by 90 degrees
plt0.xticks(rotation=90)

# Set the y-axis label
plt0.ylabel('Population < 1 million')

# Set the plot title
plt0.title('Population in urban agglomerations of more than 1 million ' +
           '(% of total population) by Country')

# Display the plot
plt0.show()


# Create a list of countries to plot
countries = ['Australia', 'United States', 'United Kingdom', 'World']

# Filter the urban_agglo DataFrame to only include rows where the
# "Country Name" column is in the countries list
urban_agglo = urban_agglo[urban_agglo['Country Name'].isin(countries)]

# Create a step plot of the data
plt0.step(urban_agglo['Country Name'], urban_agglo['2015'])

# Rotate the x-axis labels by 90 degrees
plt0.xticks(rotation=90)

# Set the y-axis label
plt0.ylabel('Population < 1 million')

# Set the plot title
plt0.title('Population in urban agglomerations of more than 1 million ' +
           '(% of total population) by Country')

# Display the plot
plt0.show()
# Call the attribute_function with the arguments "Urban population living in
#areas where elevation is below 5 meters (% of total population)" and
#df_country2
urban_pop_elevation = attribute_function("Urban population living in areas " +
                                         "where elevation is below 5 meters" +
                                         " (% of total population)",
                                         df_country2)

# Assign the resulting DataFrame to the variable urban_pop_elevation
urban_pop_elevation


# Create a list of countries to plot
countries = ['Australia', 'United States', 'United Kingdom', 'World']

# Filter the urban_pop_elevation DataFrame to only include rows where the
#"Country Name" column is in the countries list
urban_pop_elevation = urban_pop_elevation[urban_pop_elevation[
    'Country Name'].isin(countries)]

# Create a bar plot of the data
plt1.bar(urban_pop_elevation['Country Name'],
         urban_pop_elevation['2015'])

# Rotate the x-axis labels by 90 degrees
plt1.xticks(rotation=90)

# Set the y-axis label
plt1.ylabel('Urban population (% of total population')

# Set the plot title
plt1.title('Urban population living in areas where elevation is below 5\
           meters by Country')

# Display the plot
plt1.show()
# Call the attribute_function with the arguments "Rural population living in
#areas where elevation is below 5 meters (% of total population)" and
#df_country2
rural_pop_elevation = attribute_function("Rural population living in "+
                                         "areas where elevation is below 5 " + 
                                         "meters (% of total population)",
                                         df_country2)

# Assign the resulting DataFrame to the variable rural_pop_elevation
rural_pop_elevation


# Select the countries to plot
countries = ['Australia', 'United States', 'United Kingdom', 'World']

# Filter the data to include only the selected countries
rural_pop_elevation = rural_pop_elevation[rural_pop_elevation[
    'Country Name'].isin(countries)]

# Create a stem plot of the rural population living in areas where elevation is
# below 5 meters by country
plt.stem(rural_pop_elevation['Country Name'], rural_pop_elevation['2015'])

# Rotate the x-axis labels by 90 degrees
plt.xticks(rotation=90)

# Add a label to the y-axis
plt.ylabel('Rural population (% of total population')

# Add a title to the plot
plt.title('Rural population living in areas where elevation is below '+
          '5 meters by Country')

# Display the plot
plt.show()

# Read the World Bank data from a CSV file
df3 = pd.read_csv('WorldBank.csv', skiprows=4)

# Subset the data to only include GDP and Renewable Energy Consumption for a
#few countries and the world
countries = ['Australia', 'United States', 'United Kingdom', 'Argentina']
indicators = ['GDP (current US$)', 'Renewable energy consumption ' +
              '(% of total final energy consumption)']
df = df3[df3['Country Name'].isin(countries) & df3[
    'Indicator Name'].isin(indicators)]

# Group the data by country and compute the mean
df_mean = df.groupby('Indicator Name').mean().reset_index()

# Print the result
df_mean

# Read the World Bank data from a CSV file
df3 = pd.read_csv('WorldBank.csv', skiprows=4)

# Subset the data to only include GDP and Renewable Energy Consumption for a
#few countries and the world
countries = ['Australia', 'United States', 'United Kingdom', 'Argentina']
indicators = ['GDP (current US$)', 'Renewable energy consumption ' +
              '(% of total final energy consumption)']
df = df3[df3['Country Name'].isin(countries) & df3[
    'Indicator Name'].isin(indicators)]

# Group the data by country and compute the mean
df_mean = df.groupby('Indicator Name').mean().reset_index()

# Set the figure size
plt.figure(figsize=(10, 6))

# Create a line chart for GDP
gdp_data = df_mean[df_mean['Indicator Name'] == 'GDP (current US$)']
plt.plot(df['Country Name'], marker='o', label='GDP')

# Set the axis labels and title
plt.xlabel('Country')
plt.ylabel('Value')
plt.title('GDP and Renewable Energy Consumption by Country')

# Add a legend
plt.legend()

# Display the chart
plt.show()


#reading groupby dataframe
groupby_df =  read_worldbank_data('WorldBank.csv')
#calculating the mean and median of new_df
new_df = groupby_df[[ 'Country Name','1950', '1960', '1970', '1980', '1990']]
#by country name calculating mean and median for worldbank data
group_by_countries_mean = new_df.groupby('country name').mean()
group_by_countries_median = new_df.groupby('country name').median()

#printing the values of mean and median of dataframe
print(group_by_countries_mean)
print(group_by_countries_median)






