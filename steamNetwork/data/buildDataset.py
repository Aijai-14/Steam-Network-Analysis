from __future__ import annotations
import time, json, urllib.parse, requests
from pathlib import Path
import pandas as pd
from typing import Iterable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Academic Project"} 
ITAD_API = "https://api.isthereanydeal.com" # ITAD API endpoint

API_KEY = "Here is ITAD API key"


def fetch_reviews(appid: int, max_pages: int | None = None, sleep=0.5) -> pd.DataFrame:
    """Pull paginated Steam Store reviews for a single appid into a tidy DataFrame.

    Args:
        appid: Steam application identifier.
        max_pages: Optional pagination cap; None fetches the entire cursor stream.
        sleep: Delay between requests to stay polite.

    Returns:
        DataFrame with user/app/review identifiers and basic metadata.
    """
    # Steam API endpoint
    url = f"https://store.steampowered.com/appreviews/{appid}"
    # Parameters for the API request
    params = dict(json=1, filter="all", language="all",
                  review_type="all", purchase_type="all",
                  num_per_page=100, day_range=1825, cursor="*")
    
    # Initialize variables
    rows, pages = [], 0
    
    while True:
        # Make the API request
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        # Iterate through the reviews
        for rv in data.get("reviews", []):
            # Append the review data to the rows list
            rows.append(dict(
                user_id=rv["author"]["steamid"],
                app_id=appid,
                review_id=rv["recommendationid"],
                timestamp=pd.to_datetime(rv["timestamp_created"], unit="s", utc=True),
                recommended=bool(rv["voted_up"]),
                votes_up=int(rv.get("votes_up", 0)),
                playtime_hours=float(rv["author"].get("playtime_forever", 0))/60.0
            ))
        pages += 1

        batch = data.get("reviews", [])

        # Stop conditions: empty page OR cursor didn’t advance
        next_cursor = data.get("cursor")
        if not batch or not next_cursor or next_cursor == params["cursor"]:
            print("No more reviews; fetched", len(rows), "total.")
            break

        params["cursor"] = next_cursor

        if max_pages and pages >= max_pages:
            print("Reached max_pages =", max_pages)
            break
        time.sleep(sleep)
    return pd.DataFrame(rows)


def fetch_reviews_for_apps(appids: list[int], max_pages: int | None = None) -> pd.DataFrame:
    """Batch wrapper that concatenates ``fetch_reviews`` results for many appids."""
    parts = []
    # Iterate through the appids
    for a in appids:
        # Fetch the reviews for the appid
        df = fetch_reviews(a, max_pages=max_pages)
        # If the DataFrame is not empty, append it to the parts list
        if not df.empty: parts.append(df)
        print("Fetched", len(df), "reviews for app_id", a)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["user_id","app_id","review_id","timestamp","recommended","votes_up","playtime_hours"]
    )


def fetch_appdetails(appids: list[int]) -> pd.DataFrame:
    """Download selected app metadata (pricing, release, genres) from Steam."""
    rows = []
    # Iterate through the appids
    for appid in appids:
        # Make the API request
        resp = requests.get("https://store.steampowered.com/api/appdetails",
                            params={"appids": appid}, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        payload = resp.json().get(str(appid), {})
        # Get the data from the payload
        data = payload.get("data") or {}
        # Get the genres from the data
        genres = "|".join([g["description"] for g in data.get("genres", [])]) if data else ""
        # Get the release date from the data
        rel_date = data.get("release_date", {}).get("date")
        # Get the price from the data
        price = data.get("price_overview", {}) or {}
        # Append the data to the rows list
        rows.append(dict(
            app_id=appid,
            title=data.get("name"),
            release_date=rel_date,
            base_price=price.get("initial")/100.0 if price.get("initial") else None,
            current_price=price.get("final")/100.0 if price.get("final") else None,
            discount_pct=price.get("discount_percent"),
            genres=genres
        ))
        time.sleep(0.2)
    return pd.DataFrame(rows)


def fetch_steamspy_owners(appids: list[int], sleep=0.8) -> pd.DataFrame:
    """Query SteamSpy for owner-count snapshots for each appid."""
    rows = []
    # Iterate through the appids
    for appid in appids:
        # Make the API request
        r = requests.get("https://steamspy.com/api.php",
                         params={"request": "appdetails", "appid": appid},
                         headers=HEADERS, timeout=30)
        r.raise_for_status()
        d = r.json()
        # Get the owners from the data
        owners = d.get("owners", "0 .. 0").split("..")
        owners_low = int(owners[0].strip().replace(",", "")) if owners else None
        owners_high = int(owners[1].strip().replace(",", "")) if len(owners) > 1 else None
        rows.append(dict(app_id=appid, owners_low=owners_low, owners_high=owners_high, snapshot_ts=pd.Timestamp.utcnow()))
        time.sleep(sleep)
    return pd.DataFrame(rows)


def _auth_headers(api_key: str) -> Dict[str, str]:
    """Return default ITAD headers; auth handled via query params."""
    return {"Accept": "application/json"}


def itad_get_shops(api_key: str, country: str = "US") -> pd.DataFrame:
    """Fetch shop catalog and return a DataFrame; use to discover Steam's shop id."""
    # Make the API request
    r = requests.get(
        f"{ITAD_API}/service/shops/v1",
        params={"country": country, "key": api_key},
        headers=_auth_headers(api_key),
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    # Normalize to DataFrame
    return pd.DataFrame(data)


def itad_find_shop_id(api_key: str, title_match: str = "Steam", country: str = "US") -> int:
    """Resolve a shop name to its numeric ITAD identifier (e.g., Steam)."""
    # Get the shops from the API
    shops = itad_get_shops(api_key, country)
    # Get the row from the shops DataFrame
    row = shops.loc[shops["title"].str.lower() == title_match.lower()]
    # If the row is empty, raise an error
    if row.empty:
        raise RuntimeError(f"Could not find shop '{title_match}'. Available: {shops['title'].tolist()[:10]}...")
    # Return the shop id
    return int(row.iloc[0]["id"])


def itad_lookup_from_steam_appids(appids: Iterable[int], api_key: str, steam_shop_id: Optional[int] = None) -> Dict[str, Optional[str]]:
    """
    Map Steam appids -> ITAD UUIDs using lookup endpoint.
    Returns dict like {"app/220": "018d9...uuid", "app/12345": None}
    """
    if steam_shop_id is None:
        steam_shop_id = itad_find_shop_id(api_key, "Steam")

    # Create the payload
    payload = [f"app/{int(a)}" for a in appids]
    # Make the API request
    r = requests.post(
        f"{ITAD_API}/lookup/id/shop/{steam_shop_id}/v1",
        json=payload,
        params={"key": api_key},
        headers=_auth_headers(api_key),
        timeout=30,
    )
    r.raise_for_status()
    # Return the JSON response
    return r.json()  # mapping str->uuid or null


def itad_price_history_for_ids(itad_ids: Iterable[str], api_key: str,
                               shops: Optional[List[int]] = None,
                               country: str = "US",
                               since_iso: Optional[str] = None) -> pd.DataFrame:
    """
    Pull price change log for each ITAD UUID.
    Returns tidy DataFrame with columns:
    ['itad_id','timestamp','shop_id','shop_name','price','regular','currency','cut']
    """
    rows = []
    headers = _auth_headers(api_key)
    if shops is None:
        shops = [itad_find_shop_id(api_key, "Steam", country)]  # default to Steam

    for itad_id in itad_ids:
        if not itad_id:
            continue
        # Create the parameters
        params = {"id": itad_id, "country": country, "shops": ",".join(map(str, shops)), "key": api_key}
        # If the since_iso is provided, add it to the parameters
        if since_iso:
            params["since"] = since_iso

        # Make the API request
        r = requests.get(f"{ITAD_API}/games/history/v2", params=params, headers=headers, timeout=30)
        r.raise_for_status()
        # Iterate through the entries
        for entry in r.json():
            # Append the entry to the rows list
            rows.append({
                "itad_id": itad_id,
                "timestamp": entry["timestamp"],
                "shop_id": entry["shop"]["id"],
                "shop_name": entry["shop"]["name"],
                "price": entry["deal"]["price"]["amount"],
                "regular": entry["deal"]["regular"]["amount"],
                "currency": entry["deal"]["price"]["currency"],
                "cut": entry["deal"].get("cut", None),
            })
        time.sleep(0.25)  

    # Create the DataFrame
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["discount_pct"] = ((df["regular"] - df["price"]) / df["regular"] * 100).round(2)
    return df


def fetch_itad_price_history_for_appids(appids: Iterable[int], api_key: str,
                                        country: str = "US", since_iso: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience wrapper: Steam appids -> ITAD UUIDs -> price history (Steam shop only by default).
    """
    # Find the Steam shop id
    steam_id = itad_find_shop_id(api_key, "Steam", country)
    # Lookup the appids from the Steam shop id
    mapping = itad_lookup_from_steam_appids(appids, api_key, steam_id)
    # Create a dictionary of appids to ITAD ids
    app_to_itad = {int(k.split("/",1)[1]): v for k, v in mapping.items()}
    # Get the price history for the appids
    df = itad_price_history_for_ids([v for v in app_to_itad.values() if v], api_key, shops=[steam_id], country=country, since_iso=since_iso)
    # If the DataFrame is not empty, annotate the original steam appids
    if not df.empty:
        # annotate original steam appids by reverse-joining via first shop lookup if needed
        df["app_id"] = None
        rev = {v: k for k, v in app_to_itad.items() if v}
        df["app_id"] = df["itad_id"].map(rev)
        df = df[["timestamp","app_id","itad_id","shop_id","shop_name","price","regular","currency","discount_pct","cut"]].sort_values(["app_id","timestamp"])
    return df


def save_history_csv(df: pd.DataFrame, path: str) -> None:
    """Persist a cleaned price-history DataFrame."""
    df.to_csv(path, index=False)


def write_all(appids: list[int], itad_api_key: str | None = None, max_review_pages: int | None = None):
    """Top-level helper to fetch raw assets and store them under ``data/raw``."""
    # Fetch the reviews
    print("Fetching reviews…"); reviews = fetch_reviews_for_apps(appids, max_pages=max_review_pages)
    # Save the reviews to a CSV file
    reviews.to_csv(RAW / "reviews.csv", index=False)
    
    # Fetch the appdetails
    print("Fetching appdetails…"); meta = fetch_appdetails(appids)
    # Save the appdetails to a CSV file
    meta.to_csv(RAW / "games.csv", index=False)
    
    # Fetch the SteamSpy owners snapshot
    print("Fetching SteamSpy owners snapshot…"); owners = fetch_steamspy_owners(appids)
    # Save the SteamSpy owners snapshot to a CSV file
    owners.to_csv(RAW / "ownership.csv", index=False)

    # Price events from ITAD
    if itad_api_key:
        # Fetch the price history from ITAD
        hist = fetch_itad_price_history_for_appids(appids, itad_api_key, country="US", since_iso="2015-01-01T00:00:00Z")
        # Save the price history to a CSV file
        save_history_csv(hist, f"{RAW}/price_events_itad.csv")

        print("Wrote price_events_itad.csv.")


if __name__ == "__main__":
    # read app ids from data/raw/appids.txt (one per line)
    appids_txt = RAW / "appids.txt"
    if appids_txt.exists():
        appids = [int(x.strip()) for x in appids_txt.read_text().splitlines() if x.strip()]
    write_all(appids, itad_api_key=API_KEY, max_review_pages=35)
