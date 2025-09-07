import pandas as pd

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def main():
    for year in [2024, 2025]:
        clean_data(year)

def clean_data(year):
    data = load_data(f'data/all_pitches_{year}.csv')

    print(f"Filtering: {len(data)} rows from {year}...")

    # filter to rows that have a date on or after year-04-01
    # further filter to rows that have non-null values for release_speed, stand, p_throws, plate_x, plate_z, balls, strikes, type, sz_top, sz_bot
    data = data[pd.to_datetime(data['game_date']) >= pd.to_datetime(f'{year}-04-01')]
    data = data[
        data['release_speed'].notnull() &
        data['stand'].notnull() &
        data['p_throws'].notnull() &
        data['plate_x'].notnull() &
        data['plate_z'].notnull() &
        data['balls'].notnull() &
        data['strikes'].notnull() &
        data['type'].notnull() &
        data['sz_top'].notnull() &
        data['sz_bot'].notnull() &
        data['pfx_z'].notnull() &
        data['release_pos_z'].notnull() &
        data['release_pos_y'].notnull() &
        data['release_pos_x'].notnull() &
        data['release_spin_rate'].notnull()
    ]

    # save to new csv
    data.to_csv(f'data/filtered_pitches_{year}.csv', index=False)
    print(f"Filtered:  {len(data)} rows and saved to data/filtered_pitches_{year}.csv")

main()
