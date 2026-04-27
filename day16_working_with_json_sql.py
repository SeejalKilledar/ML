import pandas as pd

# working with JSON
df = pd.read_json('test.json')
print(df)

# load json from URL
print(pd.read_json('https://api.exchangerate-api.com/v4/latest/INR'))


# working with SQL
"""
1. Download xammp software from internet and install it
2. Launch xammp
3. Start Apache(Web server) and MySQL(DB server)
4. Enxter localhost/phpmyadmin/ in chrome
5. Click on New
6. Give DB name, World example and click on create
7. Click on Import
8. Click on Choose file
9. Select file world.sql (which is downloaded from kaggle)
10. Scroll down and click on import
11. Wait until you get a message Import has been successfully finished, 5344 queries executed. (world.sql)
12. Now convert the sql in xammp into pandas Dataframe
13. For the step 12, connect sql with python, install library mysql.connector
"""

# 14.
import mysql.connector

# 15. Create connection to mysql via python
conn =mysql.connector.connect(host = 'localhost', user = 'root', password = '', database = 'world')
print(pd.read_sql_query("SELECT * FROM city", conn))

