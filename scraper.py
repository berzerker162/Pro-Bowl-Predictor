#scrape NFL combine data from https://www.pro-football-reference.com/years/
# importing the libraries
# need to add a year column
import requests
import csv
from bs4 import BeautifulSoup
import numpy as np


range = np.arange(2000,2020,1) #for all years in question

outfile = open("probowlers.csv","w",newline='')
writer = csv.writer(outfile)

# Make a GET request to fetch the raw HTML content
for item in range:
    url = 'https://www.pro-football-reference.com/years/' + item.astype(str) + '/probowl.htm'
    html_content = requests.get(url).text

    tree = BeautifulSoup(html_content,"lxml")
    table_tag = tree.select("table")[0]
    tab_data = [[item.text for item in row_data.select("th,td")]
                    for row_data in table_tag.select("tr")]

    for data in tab_data:
        writer.writerow(data)
        print(' '.join(data))