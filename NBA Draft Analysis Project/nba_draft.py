import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

#importing nba draft basketball reference data and organizing it into one df
def load_data(directory):
    
    dfs = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            dfs.append(df)
    
    return dfs
        
b_ref_data = load_data('/Users/ethanwang17/Desktop/career/nba_draft/nba_stats')
b_ref_col = ['rank', 'pick', 'team', 'name', 'college', 'years', 'G_nba', 'MP_nba',
              'PTS_nba', 'TRB_nba', 'AST_nba', 'FG%_nba', '3P%_nba', 'FT%_nba', 'MP_avg_nba', 'PTS_avg_nba', 'TRB_avg_nba',
              'AST_avg_nba', 'WS_nba', 'WS/48_nba', 'BPM_nba', 'VORP_nba']

for df in b_ref_data:
    df.columns = b_ref_col

nba_df = pd.concat(b_ref_data, axis = 0, ignore_index = True)
nba_df = nba_df.dropna(how = 'all')
nba_df.sort_values(by = 'rank', inplace = True)
pd.options.mode.chained_assignment = None
nba_df.reset_index(drop = True, inplace = True)


#importing and organizing scraped cbb draft reference data
pergame_df = pd.read_csv('/Users/ethanwang17/Desktop/career/nba_draft/pergame_df.csv', encoding='latin1')
totals_df = pd.read_csv('/Users/ethanwang17/Desktop/career/nba_draft/totals_df.csv', encoding='latin1')
profiles_df = pd.read_csv('/Users/ethanwang17/Desktop/career/nba_draft/profiles_df.csv', encoding='latin1')

pergame_df[['conf', 'class']] = pergame_df[['conf', 'class']].ffill()
totals_df[['conf', 'class']] = totals_df[['conf', 'class']].ffill()

col_career_avg = pergame_df[pergame_df['season'] == 'Career']
col_career_totals = totals_df[totals_df['season'] == 'Career']

three_p = ['3P', '3PA', '3P%']
three_p2 = ['3P_per', '3PA_per', '3P%_per']
for ind, row in col_career_avg.iterrows():
    col_career_avg.loc[ind, three_p2] = row[three_p2].fillna(0)
    
    if np.isnan(col_career_avg.loc[ind, 'ORB_per']):
        col_career_avg.loc[ind, 'ORB_per'] = row['TRB_per'] * 0.30
        
    if np.isnan(col_career_avg.loc[ind, 'DRB_per']):
        col_career_avg.loc[ind, 'DRB_per'] = row['TRB_per'] * 0.70
        
for ind, row in col_career_totals.iterrows():
    col_career_totals.loc[ind, three_p] = row[three_p].fillna(0)
    
    if np.isnan(col_career_totals.loc[ind, 'ORB']):
        col_career_totals.loc[ind, 'ORB'] = row['TRB'] * 0.30
        
    if np.isnan(col_career_totals.loc[ind, 'DRB']):
        col_career_totals.loc[ind, 'DRB'] = row['TRB'] * 0.70

col_career_avg['class'] = col_career_avg['class'].astype(str)
col_career_totals['class'] = col_career_totals['class'].astype(str)

classes = ['FR', 'SO', 'JR', 'SR']
for ind, row in col_career_avg.iterrows():
    if row['class'] not in classes:
        class_index = row.index.get_loc('class')

        for i in range(len(row) - 1, class_index, -1):
            col_career_avg.at[ind, col_career_avg.columns[i]] = col_career_avg.at[ind, col_career_avg.columns[i - 1]]
        
for ind, row in col_career_totals.iterrows():
    if row['class'] not in classes:
        class_index = row.index.get_loc('class')

        for i in range(len(row) - 1, class_index, -1):
            col_career_totals.at[ind, col_career_totals.columns[i]] = col_career_totals.at[ind, col_career_totals.columns[i - 1]]
                      
col_career_avg.drop(columns = ['season', 'class', 'G', 'GS'], inplace = True)
col_career_totals.drop(columns = ['season', 'conf', 'school', 'class'], inplace = True)
nba_df.drop(columns = 'college', inplace = True)


#merging all 3 dataframes to have nba stats, college per game stats, and college totals together
df = pd.merge(nba_df, col_career_avg, on = 'name', how = 'inner')
df = pd.merge(df, col_career_totals, on = 'name', how = 'inner')
df = pd.merge(df, profiles_df, on = 'name', how = 'inner')
    
df['height'] = df['height'].astype(str)

def feet_to_inches(height):
    feet, inches = map(int, height.split('.'))
    return (feet * 12) + inches

def conv_weights(weight):
    num = weight[:-2]
    return int(num)
    
df.fillna(0, inplace = True)
df['height'] = df['height'].apply(feet_to_inches)
df['weight'] = df['weight'].apply(conv_weights)
df.drop(columns = 'rank', inplace = True)
df.loc[(df['weight'] > 215) & (df['height'] == 73), 'height'] += 9
df.drop_duplicates(inplace = True)
df.reset_index(inplace = True, drop = True)


#starting data visualization
#histograms of minutes played and games played
games = df[df['G_nba'] != 0]['G_nba'].tolist()
minutes = df[df['MP_nba'] != 0]['MP_nba'].tolist()

fig, axes = plt.subplots(1, 2, figsize=(12, 8))

fig.set_facecolor('lightgrey')
sns.histplot(games, bins=10, color='skyblue', edgecolor='black', ax=axes[0])
axes[0].set_title('Games Played in the NBA\n(2nd Round Picks)', fontsize=16)
axes[0].set_xlabel('NBA Games Played', fontsize=14)
axes[0].set_ylabel('Frequency', fontsize=14)
axes[0].set_xticks(np.arange(0, 1375, 125))
axes[0].tick_params(axis = 'x', rotation=45)

sns.histplot(minutes, bins=10, color='skyblue', edgecolor='black', ax=axes[1])
axes[1].set_title('Minutes Played in the NBA\n(2nd Round Picks)', fontsize=16)
axes[1].set_xlabel('NBA Minutes Played', fontsize=14)
axes[1].set_ylabel('Frequency', fontsize=14)
axes[1].set_xticks(np.arange(0, 36300, 3300))
axes[1].tick_params(axis = 'x', rotation=45)

plt.tight_layout()
plt.show()

#table calculations
winshares = df['WS_nba'].tolist()
vorp = df['VORP_nba'].tolist()
zero_games = len(df) - len(games)
zero_mins = len(df) - len(minutes)

#kde plots for height and weight of busts
busts = df[df['MP_nba'] < 1000]
bust_heights = busts['height']
bust_weights = busts['weight']

fig, axes = plt.subplots(1, 2, figsize=(18, 10))

fig.set_facecolor('lightgrey')
sns.kdeplot(bust_heights, fill=True, bw_adjust=0.5, ax=axes[0], color='red')
axes[0].set_title('"Complete Busts"\nHeight Distribution', fontsize=16)
mean_height = bust_heights.mean()     
axes[0].axvline(mean_height, color="green", linestyle="--", label=f"Mean: {mean_height:.2f}") 
axes[0].set_xlabel('Height', fontsize=14)
axes[0].set_ylabel('Probability Density', fontsize=14)
axes[0].legend(fontsize=14)  

sns.kdeplot(bust_weights, fill=True, bw_adjust=0.5, ax=axes[1], color='blue')
axes[1].set_title('"Complete Busts"\nWeight Distribution', fontsize=16)
mean_weight = bust_weights.mean()     
axes[1].axvline(mean_weight, color="green", linestyle="--", label=f"Mean: {mean_weight:.2f}")  
axes[1].set_xlabel('Weight', fontsize=14)
axes[1].set_ylabel('Probability Density', fontsize=14)
axes[1].legend(fontsize=14)  

plt.tight_layout()
plt.show()

#plotting percentile difference of busts at each position
g = df[df['height'] <= 77]
guards = g.loc[:, 'MP_per':].astype(float)

f = df[(df['height'] > 77) & (df['height'] <= 81)]
forwards = f.loc[:, 'MP_per':].astype(float)

c = df[df['height'] > 81]
centers = c.loc[:, 'MP_per':].astype(float)

#means
g_mean = np.mean(guards, axis=0)
g_mean = g_mean.reset_index(drop=False).T
g_mean.columns = list(g_mean.iloc[0, :])
g_mean.drop(g_mean.index[0], inplace=True)

f_mean = np.mean(forwards, axis=0)
f_mean = f_mean.reset_index(drop=False).T
f_mean.columns = list(f_mean.iloc[0, :])
f_mean.drop(f_mean.index[0], inplace=True)

c_mean = np.mean(centers, axis=0)
c_mean = c_mean.reset_index(drop=False).T
c_mean.columns = list(c_mean.iloc[0, :])
c_mean.drop(c_mean.index[0], inplace=True)

#standard deviations
g_std = np.std(guards, axis=0)
g_std = g_std.reset_index(drop=False).T
g_std.columns = list(g_std.iloc[0, :])
g_std.drop(g_std.index[0], inplace=True)

f_std = np.std(forwards, axis=0)
f_std = f_std.reset_index(drop=False).T
f_std.columns = list(f_std.iloc[0, :])
f_std.drop(f_std.index[0], inplace=True)

c_std = np.std(centers, axis=0)
c_std = c_std.reset_index(drop=False).T
c_std.columns = list(c_std.iloc[0, :])
c_std.drop(c_std.index[0], inplace=True)

#positonal busts
g_busts = busts[busts['height'] <= 77]
g_busts = g_busts.loc[:, 'MP_per':].astype(float)

f_busts = busts[(busts['height'] > 77) & busts['height'] <= 81]
f_busts = f_busts.loc[:, 'MP_per':].astype(float)

c_busts = busts[busts['height'] > 81]
c_busts = c_busts.loc[:, 'MP_per':].astype(float)

#mse for each position
g_perc = []
for i in range(g_busts.shape[1]):
    column_data = list(g_busts.iloc[:, i])
    column_mean = g_mean.iloc[:, i]
    perc = []
    for value in column_data:
        percentile = ((value - column_mean) / column_mean) * 100
        perc.append(percentile.values[0])
    
    g_perc.append(perc)
    
f_perc = []
for i in range(f_busts.shape[1]):
    column_data = list(f_busts.iloc[:, i])
    column_mean = f_mean.iloc[:, i]
    perc2 = []
    for value in column_data:
        percentile = ((value - column_mean) / column_mean) * 100
        perc2.append(percentile.values[0])
    
    f_perc.append(perc2)

c_perc = []
for i in range(c_busts.shape[1]):
    column_data = list(c_busts.iloc[:, i])
    column_mean = c_mean.iloc[:, i]
    perc3 = []
    for value in column_data:
        percentile = ((value - column_mean) / column_mean) * 100
        perc3.append(percentile.values[0])
    
    c_perc.append(perc3)

perc_col = ['MP', 'FG', 'FGA', 'FG%', '2P', '2PA',
       '2P%', '3P', '3PA', '3P%', 'FT', 'FTA',
       'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
       'BLK', 'TOV', 'PF', 'PTS', 'SOS', 'G', 'GS', 'MP', 'FG',
       'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT', 'FTA',
       'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
       'height', 'weight']

guards_perc = pd.DataFrame(g_perc).T
guards_perc.columns = perc_col

forwards_perc = pd.DataFrame(f_perc).T
forwards_perc.columns = perc_col

centers_perc = pd.DataFrame(c_perc).T
centers_perc.columns = perc_col

g_final = guards_perc.mean()
f_final = forwards_perc.mean()
c_final = centers_perc.mean()

#dfs for percentile mean
g_bust_analysis = pd.DataFrame(g_final).T
g_bust_analysis.drop(columns = ['height', 'weight'], inplace=True)
f_bust_analysis = pd.DataFrame(f_final).T
f_bust_analysis.drop(columns = ['height', 'weight'], inplace=True)
c_bust_analysis = pd.DataFrame(c_final).T
c_bust_analysis.drop(columns = ['height', 'weight'], inplace=True)

g_bust_per = g_bust_analysis.iloc[0, :23].to_frame()
g_bust_per.reset_index(drop=False, inplace=True)
g_bust_tot = g_bust_analysis.iloc[0, 23:].to_frame()
g_bust_tot.reset_index(drop=False, inplace=True)

f_bust_per = f_bust_analysis.iloc[0, :23].to_frame()
f_bust_per.reset_index(drop=False, inplace=True)
f_bust_tot = f_bust_analysis.iloc[0, 23:].to_frame()
f_bust_tot.reset_index(drop=False, inplace=True)

c_bust_per = c_bust_analysis.iloc[0, :23].to_frame()
c_bust_per.reset_index(drop=False, inplace=True)
c_bust_tot = c_bust_analysis.iloc[0, 23:].to_frame()
c_bust_tot.reset_index(drop=False, inplace=True)

subplot_info = [{'title': '2nd Round Guards', 'ylabel': '% Deviation'},
                {'title': '2nd Round Forwards', 'ylabel': '% Deviation'},
                {'title': '2nd Round Centers', 'ylabel': '% Deviation'}]
colors = ['#8FBC8F', '#333333', '#9370DB']  

with sns.axes_style("whitegrid"): 
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor='lightgrey')
    
for i, (df, ax, info) in enumerate(zip([g_bust_per, f_bust_per, c_bust_per], axes, subplot_info)):
    x = df['index']
    y = df.iloc[:, 1]
    ax.bar(x, y, color=colors[i])
    ax.set_title(info['title'], fontsize = 15)
    ax.set_ylabel(info['ylabel'])
    ax.set_xticklabels(x, rotation=90)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')  
        spine.set_linewidth(1) 
 
plt.suptitle('% Deviation from the Mean\n(per game averages)', 
             fontsize=18,
             fontweight='bold')
plt.tight_layout()
plt.show()


with sns.axes_style("whitegrid"):  
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor='lightgrey')
    
for i, (df, ax, info) in enumerate(zip([g_bust_tot, f_bust_tot, c_bust_tot], axes, subplot_info)):
    x = df['index']
    y = df.iloc[:, 1]
    ax.bar(x, y, color=colors[i])
    ax.set_title(info['title'], fontsize=15)
    ax.set_ylabel(info['ylabel'])
    ax.set_xticklabels(x, rotation=90)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')  
        spine.set_linewidth(1) 
        
plt.suptitle('% Deviation from the Mean\n(career totals)', 
             fontsize=18, 
             fontweight='bold')
plt.tight_layout()
plt.show()




