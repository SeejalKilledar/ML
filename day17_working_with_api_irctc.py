import pandas as pd
import requests
#
# url = "https://booking-com15.p.rapidapi.com/api/v1/meta/locationToLatLong"
#
# # querystring = {"query":"man"}
#
# headers = {
# 	"x-rapidapi-key": "484d7fd80cmshd901efff94ddc55p185d41jsnc1db465119cd",
# 	"x-rapidapi-host": "booking-com15.p.rapidapi.com"
# }
#
# response = requests.get(url, headers=headers)
#
# # val = response.json()
# # print(val['data'])
#
# # df = (pd.DataFrame(response.json()['data']))
# # print(df.columns.tolist())
#
# df = (pd.DataFrame(response.json()['data']).head()[['business_status', 'formatted_address', 'geometry', 'name', 'opening_hours', 'photos', 'place_id', 'price_level', 'rating', 'reference', 'types', 'user_ratings_total']])
# df.to_csv('booking_com.csv')




all_data = []
df = pd.DataFrame()

for i in range(1,17):
	url = "https://booking-com15.p.rapidapi.com/api/v1/meta/locationToLatLong"
	headers = {
		"x-rapidapi-key": "484d7fd80cmshd901efff94ddc55p185d41jsnc1db465119cd",
		"x-rapidapi-host": "booking-com15.p.rapidapi.com"
	}

	querystring = {"query": "man"}
	response = requests.get(url, headers=headers, params=querystring)
	temp_df = (pd.DataFrame(response.json()['data']).head()[
		['business_status', 'formatted_address', 'geometry', 'name', 'opening_hours', 'photos', 'place_id',
		 'price_level', 'rating', 'reference', 'types', 'user_ratings_total']])
	all_data.append(temp_df)

df = pd.concat(all_data, ignore_index=True)
df.to_csv('booking_com.csv', index=False)