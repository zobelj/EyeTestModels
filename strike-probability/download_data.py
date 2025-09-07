import requests as r
import pandas as pd
from io import StringIO

def download_team_pitches(team_id):
    url = f"https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfBBT=&hfPR=ball%7Cblocked%5C.%5C.ball%7Ccalled%5C.%5C.strike%7C&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=R%7CPO%7CS%7C=&hfSea=2024|&hfSit=&player_type=pitcher&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&team={team_id}&position=&hfRO=&home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc&min_abs=0&type=details"
    response = r.get(url)

    if response.status_code == 200:
        print(f" --> team ID: {team_id}")
        return pd.read_csv(StringIO(response.text))
    else:
        raise Exception(f"Failed to download data for team ID {team_id}: {response.status_code}")

def download_all():
    print("Starting download of all team data...")
    teams = pd.read_json("data/teams.json")

    dfs_list = []
    for team_id in teams["mlbam_id"]:
        team_pitches = download_team_pitches(team_id)

        team_pitches["team_id"] = team_id
        dfs_list.append(team_pitches)

    df = pd.concat(dfs_list, ignore_index=True)

    df.to_csv("data/all_pitches_2024.csv", index=False)
    print("All data downloaded and saved to data/all_pitches.csv")

if __name__ == "__main__":
    download_all()
