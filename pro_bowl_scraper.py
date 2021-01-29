#Get list of all NFL pro bowlers from https://www.pro-football-reference.com/years/
import csv
import numpy as np
import requests
from bs4 import BeautifulSoup


#update for all years available
range = np.arange(2000,2020,1) #for all years in question

outfile = open("probowlers_new.csv","w",newline='')
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