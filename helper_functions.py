import numpy as np
import pandas as pd


def preprocess_weather(df, moving_avg_duration_list):
    # moving_avg_duration should be in the form ['2d', '7d', '14d']
    # if windgust > 0, then 1, else 0. Remove scalar values since they are all within a very narrow range.
    df_ = df.copy()
    df_['wind_gust'] = df_['wind_gust'].apply(lambda x: 1 if x > 0 else 0)

    # create 3n new columns where n is the numerical columns in the data-frame. These columns will be the moving
    # averages of these values.

    features_numerical = ['pressure_station', 'pressure_sea', 'wind_dir_10s', 'wind_speed',
                          'relative_humidity', 'dew_point', 'temperature']

    df_.set_index('date_time_local', inplace=True)
    for feature in features_numerical:
        for duration in moving_avg_duration_list:
            df_[f'{feature}_{duration}'] = df_[feature].rolling(duration).mean()

    # TODO: add seasonality feature - ordinal?

    df_.sort_index(inplace=True)
    return df_


def clean_weather_df(df):
    # Parse dates correctly
    df['date_time_local'] = pd.to_datetime(df['unixtime'], unit='s')
    # Drop un-needed columns:
    columns_to_drop = ['unixtime', 'wind_dir', 'windchill', 'humidex', 'visibility', 'health_index', 'cloud_cover_4',
                       'cloud_cover_8', 'cloud_cover_10', 'solar_radiation']
    df_ = df.drop(columns=columns_to_drop)
    return df_


def duration_calculator(end_time_np_array, duration):
    """
    Creates a nx2 numpy array where n is the number of dates. The array contains the start and end dates based
    on the duration provided.
    :param end_time_np_array: numpy array of datetime objects to be used as the end time of the duration.
    :param duration: integer indicating the number of days you want to go back from start date
    :return: nx2 numpy array with the following structure:
    [[start_date, end_date],
    [start_date, end_date],
    ...
    [start_date, end_date]]
    """
    time_delta = pd.Timedelta(value=duration, unit='days')
    time_start = end_time_np_array - time_delta
    duration = np.stack((time_start, end_time_np_array), axis=-1)

    return duration


def clean_mosquito_df(df, gender=None):
    """
    Pre-processes the mosquito df and returns a df with the index set as the date and the count column.
    If gender is not specified, it combines the male and female counts of the mosquitoes
    :param df: mosquito dataframe
    :param gender: 'Male', 'Female' (optional)
    :return: df with index as date and counts. All other columns are removed.
    """
    # Parse dates:
    df['Trap Date'] = pd.to_datetime(df['Trap Date'])

    # Remove unneeded columns
    mosquito_df_ = df.drop(columns=['Comparison Group', 'Genus',
                                    'Specific Epithet', 'Trap Region',
                                    'Latitude', 'Longitude',
                                    'Location', 'IDd'])

    # Remove all rows marked as No or No Data in the Include column
    mosquito_df_ = mosquito_df_.loc[mosquito_df_['Include'].isnull()]

    # Drop the 'Include' column
    mosquito_df_.drop(columns=['Include'], inplace=True)

    # Subgroup dictates if we want the male, female or combined data set
    # We assume that the all locations are counted each week however only the locations with count > 0 show up.
    # As such we will sum up all counts for locations.
    if gender:
        mosquito_df_ = mosquito_df_.loc[mosquito_df_['Gender'] == gender]
    mosquito_df_ = mosquito_df_.groupby(by=['Trap Date']).sum()
    mosquito_df_.sort_index(inplace=True)
    return mosquito_df_


def merge_mosquito_weather_data(m_df, w_df):
    merged_df = pd.merge_asof(m_df,
                              w_df,
                              left_index=True,
                              right_index=True,
                              direction='nearest',
                              tolerance=pd.Timedelta(value=1, unit='days'))

    # remove all rows where the weather data is missing:
    merged_df = merged_df.loc[~merged_df['pressure_station'].isnull()]

    return merged_df
