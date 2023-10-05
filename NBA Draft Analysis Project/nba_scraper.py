import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
import time
from bs4 import BeautifulSoup
import re


#importing basketball reference data and organizing it into one df
def load_data(directory):
    
    dfs = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            dfs.append(df)
    
    return dfs
        
b_ref_data = load_data('/Users/ethanwang17/Desktop/career/nba_draft/nba_stats')
b_ref_col = ['rank', 'pick', 'team', 'player', 'college', 'years', 'G', 'MP',
              'PTS', 'TRB', 'AST', 'FG%', '3P%', 'FT%', 'MP_avg', 'PTS_avg', 'TRB_avg',
              'AST_avg', 'WS', 'WS/48', 'BPM', 'VORP']

for df in b_ref_data:
    df.columns = b_ref_col

b_ref = pd.concat(b_ref_data, axis = 0, ignore_index = True)
b_ref = b_ref.dropna(how = 'all')
b_ref.sort_values(by = 'rank', inplace = True)
pd.options.mode.chained_assignment = None
b_ref.reset_index(drop = True, inplace = True)


#scraping college stats and height, weight off of cbb sports reference 
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--disable-gpu') 
chrome_options.add_argument('--disable-extensions')
chrome_options.add_argument('--disable-infobars')
chrome_options.add_argument('--disable-dev-shm-usage') 
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--blink-settings=imagesEnabled=false')

def scrape_htmls():
    
    """scraping the cbb reference website to get htmls for every player 
    drafted in our dataframe imported above"""
    
    player_list = list(b_ref['player'])    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), 
                              options=chrome_options)
    driver.get('https://www.sports-reference.com/cbb/')
    
    player_htmls = []
    for player in player_list:

        search = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//input[@class='ac-input completely']")))
        search.send_keys(player)

        try:
            search.send_keys(Keys.RETURN)
            WebDriverWait(driver, 2).until(lambda driver: '/players/' in driver.current_url)
            time.sleep(1)
            profile_html = driver.page_source
            
        except TimeoutException:
            profile_html = None
        
        player_htmls.append(profile_html)
        time.sleep(1)
        driver.get('https://www.sports-reference.com/cbb/')

    driver.quit()

    return player_htmls

html_dic = scrape_htmls()


#creating lists of profile information to enter as columns in the new dataframe
round_2 = [item for item in html_dic if item is not None]
profiles = []

for html in round_2:
    soup = BeautifulSoup(html, 'html.parser')
    profile = soup.find('div', id="info")
        
    name = profile.find('span').text
    anchor = profile.find('a', href=lambda href: href and '/schools/' in href)
    school = anchor.get_text()
    school = school[:-6]
        
    pattern = re.compile(r'\d-\d')
    height = profile.find('span', string=pattern)
    
    if height:
        weight = height.find_next_sibling()
    
    if height and weight:
        profiles.append((name, school, height.text, weight.text))
    
            
#creating tuples for the college stats data to put in the new dataframe
pergame = []
totals = []

for html in round_2:
    soup = BeautifulSoup(html, 'html.parser')
    table1 = soup.find('table', id="players_per_game")
    table2 = soup.find('table', id="players_totals")
    profile = soup.find('div', id="info")
    name = profile.find('span').text
            
    if table1:
        pergame_rows = table1.find_all('tr')
        row_list = []
        for row in pergame_rows:
            cells = row.find_all(['td', 'th'])
            row_data = (name,) + tuple(cell.get_text(strip=True) for cell in cells)
            row_list.append(row_data)
        pergame.append(row_list)
        
    if table2:
        totals_rows = table2.find_all('tr')
        row_list2 = []
        for row in totals_rows:
            cells = row.find_all(['td', 'th'])
            row_data = (name,) + tuple(cell.get_text(strip=True) for cell in cells)
            row_list2.append(row_data)
        totals.append(row_list2)


#creating the dataframes
ind1 = []
val1 = []
for listt in pergame:
    if listt != None:
        indexes = [item[:2] for item in listt[1:]]
        ind1.append(indexes)
        values = [item[2:] for item in listt[1:]]
        val1.append(values)

flattened_1 = [item for sublist in ind1 for item in sublist]
flattened_2 = [item for sublist in val1 for item in sublist]

pergame_index = pd.MultiIndex.from_tuples(flattened_1, names=['name', 'season'])
pergame_df = pd.DataFrame(flattened_2, index=pergame_index)
pergame_df.columns = ['school', 'conf', 'class', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '2P', '2PA', '2P%',
                      '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV',
                      'PF', 'PTS', 'none', 'SOS']

ind2 = []
val2 = []
for listt in totals:
    if listt != None:
        indexes = [item[:2] for item in listt[1:]]
        ind2.append(indexes)
        values = [item[2:] for item in listt[1:]]
        val2.append(values)

flattened_3 = [item for sublist in ind2 for item in sublist]
flattened_4 = [item for sublist in val2 for item in sublist]

totals_index = pd.MultiIndex.from_tuples(flattened_3, names=['name', 'season'])
totals_df = pd.DataFrame(flattened_4, index=totals_index)
totals_df.columns = ['school', 'conf', 'class', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '2P', '2PA', '2P%',
                      '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV',
                      'PF', 'PTS']

profiles_df = pd.DataFrame(profiles, columns = ['name', 'school', 'height', 'weight'])
profiles_df['height'] = profiles_df['height'].str.replace('-', '.')

#exporting csv files to desktop
pergame_desk = '/Users/ethanwang17/Desktop/career/pergame_df.csv'
totals_desk = '/Users/ethanwang17/Desktop/career/totals_df.csv'
profiles_desk = '/Users/ethanwang17/Desktop/career/profiles_df.csv'

pergame_df.to_csv(pergame_desk)
totals_df.to_csv(totals_desk)
profiles_df.to_csv(profiles_desk)





        
        
        
        