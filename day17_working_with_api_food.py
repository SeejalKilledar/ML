import requests
import pandas as pd
import time

# API details
url = "https://tasty.p.rapidapi.com/recipes/list"
headers = {
    "x-rapidapi-key": "484d7fd80cmshd901efff94ddc55p185d41jsnc1db465119cd",
    "x-rapidapi-host": "tasty.p.rapidapi.com"
}

all_data = []
page_size = 20  # number of recipes per request
max_pages = 10  # you can increase this later (e.g., 50 or 100)

for i in range(max_pages):
    querystring = {
        "from": str(i * page_size),
        "size": str(page_size),
        "tags": "under_30_minutes"
    }
    print(f"🔹 Fetching page {i + 1} ...")
    response = requests.get(url, headers=headers, params=querystring)

    # Handle failed requests gracefully
    if response.status_code != 200:
        print(f"⚠️ Page {i + 1} failed with status {response.status_code}")
        print(response.text)
        break

    data = response.json()

    # Stop if API returns no results
    if "results" not in data or not data["results"]:
        print(f"ℹ️ No more results found at page {i + 1}. Stopping.")
        break

    temp_df = pd.json_normalize(data["results"])
    all_data.append(temp_df)

    # Respectful delay to avoid rate limits
    time.sleep(1)

# Combine all pages
if all_data:
    df = pd.concat(all_data, ignore_index=True)

    # Optional: keep only relevant columns
    useful_cols = [
        "name", "description", "country", "num_servings",
        "total_time_minutes", "user_ratings.score",
        "user_ratings.count_positive", "user_ratings.count_negative",
        "nutrition.calories", "nutrition.protein",
        "nutrition.fat", "nutrition.carbohydrates"
    ]
    df = df[[c for c in useful_cols if c in df.columns]]

    df.to_csv("tasty_recipes_all.csv", index=False, encoding="utf-8")
    print(f"\n✅ Saved {len(df)} recipes to tasty_recipes_all.csv")
else:
    print("❌ No data fetched from API.")