import pandas as pd
import requests

url = "https://tmdb-movies-and-tv-shows-api-by-apirobots.p.rapidapi.com/v1/tmdb"

querystring = {"name":"robot","page":"1"}

headers = {
	"x-rapidapi-key": "484d7fd80cmshd901efff94ddc55p185d41jsnc1db465119cd",
	"x-rapidapi-host": "tmdb-movies-and-tv-shows-api-by-apirobots.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

# data = (response.json())
# print(data['items'])
print(pd.DataFrame(response.json()['items']).head(2))
#print(pd.DataFrame(response.json()['items']).head(2)[['title', 'release_date']])


"""
if pages are 428

df = pd.Dataframe()
for i in range(1,429):
    querystring = {"name":"robot","page":i}

    headers = {
        "x-rapidapi-key": "484d7fd80cmshd901efff94ddc55p185d41jsnc1db465119cd",
        "x-rapidapi-host": "tmdb-movies-and-tv-shows-api-by-apirobots.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)    
    temp_df = pd.DataFrame(response.json()['items']).head(2)[['title', 'release_date']]
    df.append(temp_df, ignore_index = True) #ignore_index --> creating index on its own
"""