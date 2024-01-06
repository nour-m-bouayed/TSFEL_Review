from tsfel.feature_extraction.features_utils import set_domain
import pandas as pd


@set_domain("domain", "temporal")
def is_it_weekend(signal, parameters=None):
    """
    Returns 0 if the span of the time series is a weekday, 1 if it's a weekend day.

    Parameters
    ----------
    signal : pandas.Series
        The time series to calculate the feature of. Index must be a datetime type.

    Returns
    -------
    int
        0 for weekdays, 1 for weekends.
    """
    # Ensure the index is a pandas datetime type
    if not isinstance(signal.index, pd.DatetimeIndex):
        raise ValueError("Index of the signal must be a pandas DatetimeIndex.")

    # Calculate the total time span of the time series
    total_span = signal.index[-1] - signal.index[0]

    # If the total span is a day, check if it's a weekend or a weekday
    if pd.Timedelta('0.8 days')< total_span < pd.Timedelta('1.2 days'):
        # Get the day of the week (Monday=0, Sunday=6)
        day_of_week = signal.index[0].dayofweek
        
        # Return 1 if it's a weekend (Saturday or Sunday), otherwise 0
        return 1 if day_of_week >= 5 else 0
    else:
        raise ValueError("The total span of the time series is not within a single day.")

