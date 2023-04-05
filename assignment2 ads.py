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


