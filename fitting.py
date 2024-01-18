
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import errors as err

from sklearn.preprocessing import StandardScaler

def read_data(file_paths, selected_country, start_year, end_year):
    dataframes_list = []

    for path in file_paths:
        file_name = path.split('/')[-1].split('.')[0]
        df = pd.read_csv(path, skiprows=4)
        df = df.rename(columns={'Country Name': 'Country'})
        df = df.set_index('Country')
        df_selected_country = df.loc[selected_country, str(start_year):str(end_year)].transpose().reset_index()
        df_selected_country = df_selected_country.rename(columns={'index': 'Year', selected_country: file_name})
        dataframes_list.append(df_selected_country)

    result_df = pd.concat(dataframes_list, axis=1)

    result_df = result_df.apply(lambda col: col.fillna(col.mean()))

    return result_df

def exp_growth(t, scale, growth):
    """Computes exponential function with scale and growth as free parameters."""
    return scale * np.exp(growth * t)

def plot_fit(dataframe, x_col, y_col, fit_function, title, save_filename):
    """Fits the given function to the data and plots the results."""
    initial_guess = [1.0, 0.02]

    # Convert the "Year" column to numeric
    dataframe["Year"] = pd.to_numeric(dataframe["Year"], errors='coerce')
    dataframe[y_col] = pd.to_numeric(dataframe[y_col], errors='coerce')

    popt, pcovar = opt.curve_fit(fit_function, dataframe[x_col], dataframe[y_col], p0=initial_guess, maxfev=10000)

    dataframe["pop_fit"] = fit_function(dataframe[x_col], *popt)

    plt.figure()
    plt.plot(dataframe[x_col], dataframe[y_col], label="data")
    plt.plot(dataframe[x_col], dataframe["pop_fit"], label="fit")
    plt.legend()
    plt.title(title)
    plt.show()

    years = np.linspace(dataframe[x_col].min(), dataframe[x_col].max() + 5, num=1000)
    pop_fit_growth = fit_function(years, *popt)
    sigma = err.error_prop(years, fit_function, popt, pcovar)
    low = pop_fit_growth - sigma
    up = pop_fit_growth + sigma

    plt.figure()
    plt.title(title + " in 2030")
    plt.plot(dataframe[x_col], dataframe[y_col], label="data")
    plt.plot(years, pop_fit_growth, label="fit")
    plt.fill_between(years, low, up, alpha=0.3, color="y", label="95% Confidence Interval")
    plt.legend(loc="upper left")
    plt.xlabel("Year")
    plt.ylabel(y_col)
    plt.savefig(save_filename, dpi=300)
    plt.show()

    return popt, pcovar

def predict_future_values(years, fit_function, fit_parameters):
    """Predicts future values using the fitted function and parameters."""
    predictions = fit_function(years, *fit_parameters)
    return predictions / 1.0e6

# Example usage for Death Injuries
selected_country = "India"
start_year = 2000
end_year = 2019

file_paths = ['DeathInjuries.csv']
death_injuries_df = read_data(file_paths, selected_country, start_year, end_year)
death_injuries_fit_params, _ = plot_fit(death_injuries_df, "Year", "DeathInjuries", exp_growth,
                                         "Death Injuries in India", 'DeathInjuries_India.png')

# Predict future values for Death Injuries
years_to_predict = np.arange(2024, 2034)
predictions_death_injuries = predict_future_values(years_to_predict, exp_growth, death_injuries_fit_params)
for year, prediction in zip(years_to_predict, predictions_death_injuries):
    print(f"Death Injuries in India in {year}: {prediction} Mill.")

# Example usage for Urban Population in India
selected_country = "India"
start_year = 2000
end_year = 2020

file_paths = ['Urban Population.csv']
urban_population_india_df = read_data(file_paths, selected_country, start_year, end_year)
urban_population_india_fit_params, _ = plot_fit(urban_population_india_df, "Year", "Urban Population", exp_growth,
                                               "Urban Population in India", 'UrbanPopulation_India.png')

# Predict future values for Urban Population in India
predictions_urban_population_india = predict_future_values(years_to_predict, exp_growth, urban_population_india_fit_params)
for year, prediction in zip(years_to_predict, predictions_urban_population_india):
    print(f"Urban Population in India in {year}: {prediction} Mill.")

# Example usage for Urban Population in China
selected_country = "China"
start_year = 2000
end_year = 2020

urban_population_china_df = read_data(file_paths, selected_country, start_year, end_year)
urban_population_china_fit_params, _ = plot_fit(urban_population_china_df, "Year", "Urban Population", exp_growth,
                                                "Urban Population in China", 'UrbanPopulation_China.png')

# Predict future values for Urban Population in China
predictions_urban_population_china = predict_future_values(years_to_predict, exp_growth, urban_population_china_fit_params)
for year, prediction in zip(years_to_predict, predictions_urban_population_china):
    print(f"Urban Population in China in {year}: {prediction} Mill.")
