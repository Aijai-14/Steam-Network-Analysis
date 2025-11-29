from __future__ import annotations
import requests
import pandas as pd
from typing import List

STEAMSPY_API = "https://steamspy.com/api.php" # SteamSpy API endpoint


def fetch_steamspy_app_details(appids: List[int]) -> pd.DataFrame:
    """Fetch raw SteamSpy metadata rows for each appid."""
    rows = []
    # Iterate through the appids
    for appid in appids:
        params = {"request": "appdetails", "appid": int(appid)} # Create the parameters
        resp = requests.get(STEAMSPY_API, params=params, timeout=30) # Make the API request
        resp.raise_for_status()
        data = resp.json() # Get the data from the response
        data["appid"] = appid
        rows.append(data)
    return pd.DataFrame(rows)