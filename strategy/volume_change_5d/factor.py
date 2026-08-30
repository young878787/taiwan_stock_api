import pandas as pd
import numpy as np

def calculate_volume_change_5d():
    # Read the data from daily_pv.h5
    df = pd.read_hdf('daily_pv.h5')
    
    # Ensure data is sorted by datetime within each instrument
    df = df.sort_index()
    
    # Extract volume column
    volume = df['$volume']
    
    # Group by instrument and shift volume by 5 trading days to get Vol_{t-5}
    volume_lag5 = volume.groupby(level='instrument').shift(5)
    
    # Calculate the 5-day volume change factor: V^{(5)}_t = Vol_t / Vol_{t-5} - 1
    factor = volume / volume_lag5 - 1
    
    # Replace infinite values with NaN (caused by division by zero)
    factor = factor.replace([np.inf, -np.inf], np.nan)
    
    # Create result dataframe with the factor name as column
    result = pd.DataFrame({
        'volume_change_5d': factor
    })
    
    # Save result to result.h5
    result.to_hdf('result.h5', key='data')
    
    return result

if __name__ == '__main__':
    calculate_volume_change_5d()