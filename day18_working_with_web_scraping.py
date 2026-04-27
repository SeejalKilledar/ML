import pandas as pd
import requests
from bs4 import BeautifulSoup


#
# header = { 'User-Agent' :    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
# response = requests.get('https://www.ambitionbox.com/list-of-companies?page=1', headers=header).text
# soup = BeautifulSoup(response, "html.parser")
# resp= soup.find_all('div',class_ = 'companyCardWrapper')
#
# name = []
# rating = []
# review_count = []
# ctype = []
# hq = []
# ctype_hq = []
#
# for i in resp:
#     name.append(i.find('h2').text.strip())
#     rating.append(i.find('div', class_ = 'rating_text rating_text--md').get_text(strip=True))
#     review_count.append(i.find('span', class_ = 'companyCardWrapper__ActionCount').get_text(strip=True))
#     hq_list=i.find('span', class_ = 'companyCardWrapper__interLinking').get_text(strip=True)
#     #i.split('|')
#     new_hq_list = (hq_list.split("|"))
#     ctype.append(new_hq_list[0])
#     hq.append(new_hq_list[1])
# d = {
#
#     'Name' : name,
#     'Rating' : rating,
#     'Review Count' : review_count,
#     'Company Type' : ctype,
#     'Head Quarters' : hq
# }
#
# df = pd.DataFrame(d)
# print(df)


final = pd.DataFrame()

for j in range(1,11):
    url = 'https://www.ambitionbox.com/list-of-companies?page={}'.format(j)

    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=header).text

    soup = BeautifulSoup(response, "html.parser")
    resp = soup.find_all('div', class_='companyCardWrapper')

    name = []
    rating = []
    review_count = []
    ctype = []
    hq = []
    ctype_hq = []

    for i in resp:
        name.append(i.find('h2').text.strip())
        rating.append(i.find('div', class_='rating_text rating_text--md').get_text(strip=True))
        review_count.append(i.find('span', class_='companyCardWrapper__ActionCount').get_text(strip=True))
        hq_list = i.find('span', class_='companyCardWrapper__interLinking').get_text(strip=True)
        # i.split('|')
        new_hq_list = (hq_list.split("|"))
        #print(new_hq_list)
        ctype.append(new_hq_list[0])
        #hq.append(new_hq_list[1])

    d = {

        'Name': name,
        'Rating': rating,
        'Review Count': review_count,
        'Company Type': ctype
        #'Head Quarters': hq
    }

    df = pd.DataFrame(d)
    final = final._append(df,ignore_index=True)

print(final)