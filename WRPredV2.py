from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
app = Flask(__name__)

# Load your CSV data
data = pd.read_csv("C:\\Users\\pauls\\Downloads\\NFL stats\\2023an2024week3.csv")# data path

pass_data = data.dropna(subset=["air_yards"])
air_yards_by_qb_game = pass_data.groupby(['game_id', 'passer_player_name'])['air_yards'].sum().reset_index()

# Prepare pass data
pass_data = pass_data[[ 'game_id', 'week','posteam', 'defteam', "receiving_yards","pass_length", 'pass_location',
                        'yards_gained' ,'air_yards', 'yards_after_catch', 'touchdown', 'passer_player_name', 
                        'passing_yards', 'receiver_player_name', 'roof', 'wind', 'play_type_nfl']]

pass_data.drop(pass_data[pass_data.play_type_nfl == "penalty"].index, inplace = True)
pass_data['completion?'] = None
pass_data.loc[pass_data['yards_gained'] == pass_data['receiving_yards'], 'completion?'] = 1.0
pass_data.loc[pass_data['yards_gained'] != pass_data['receiving_yards'], 'completion?'] = 0.0

# Create reception info
reception_info = pass_data[['game_id', 'posteam', 'defteam', "pass_length", 'pass_location', 'air_yards', 'yards_after_catch', 'touchdown', 'passer_player_name','receiving_yards','completion?', 'receiver_player_name', 'roof', 'wind']].groupby(by = ["posteam", "defteam", 'game_id', 'receiver_player_name', 'passer_player_name'], as_index = False).sum()
reception_info.sort_values(by = ["receiver_player_name", "game_id"], inplace = True)
reception_info.reset_index(drop = True, inplace = True)

reception_info['week'] = reception_info['game_id'].str.split('_').str[1].astype(int)

# Merge with air yards by QB
reception_info = pd.merge(reception_info, air_yards_by_qb_game, on=['game_id', 'passer_player_name'], how='left')
reception_info['air_yard_share'] = reception_info['air_yards_x'] / reception_info['air_yards_y']

# Filter defensive data
df_filtered = data[['defteam', 'week', 'yards_gained', 'pass', 'yards_after_catch', 'complete_pass']].fillna({
    'yards_gained': 0, 'pass': 0, 'yards_after_catch': 0, 'complete_pass': 0
})

df_filtered['passing_yards_allowed'] = df_filtered['yards_gained'] * df_filtered['pass']
df_agg = df_filtered.groupby(['defteam', 'week']).agg(
    total_yards_allowed=pd.NamedAgg(column='yards_gained', aggfunc='sum'),
    total_receptions_allowed=pd.NamedAgg(column='complete_pass', aggfunc='sum'),
    total_passing_yards_allowed=pd.NamedAgg(column='passing_yards_allowed', aggfunc='sum'),
    total_passing_plays_allowed=pd.NamedAgg(column='pass', aggfunc='sum')
).reset_index()

# Define the home route to render the form
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/player-names', methods=['GET'])
def get_player_names():
    # Extract player names from dataframe and convert to list
    player_names = reception_info['receiver_player_name'].tolist()
    player_names = list(set(player_names))
    return jsonify(player_names)

@app.route('/defense-names', methods=['GET'])
def get_defense_names():
    # Extract player names from dataframe and convert to list
    defense_names = df_agg['defteam'].tolist()
    defense_names = list(set(defense_names))
    return jsonify(defense_names)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    playerInput = request.form['player']
    defenseInput = request.form['defense']
    
    # Filter player and defense data

    df_playerSelection = reception_info[reception_info.receiver_player_name == playerInput].iloc[-10:]
    week_counts = df_playerSelection['week'].value_counts()
    weeks_to_drop = week_counts[week_counts > 1].index
    playerSelection = df_playerSelection[~df_playerSelection['week'].isin(weeks_to_drop)]

    print(playerSelection)
    df_defSelection = df_agg[df_agg.defteam == defenseInput].iloc[-10:]
    
    # Add defense stats to player selection
    playerSelection["total_yards_allowed"] = np.nan
    playerSelection["total_receptions_allowed"] = np.nan
    playerSelection["total_passing_yards_allowed"] = np.nan
    playerSelection["total_passing_plays_allowed"] = np.nan
    
    for index, row in playerSelection.iterrows():
        for index2, row2 in df_agg.iterrows():
            if (row['defteam'] == row2['defteam']) & (row['week'] == row2['week']):
                playerSelection.loc[index, 'total_yards_allowed'] = row2['total_yards_allowed']
                playerSelection.loc[index, 'total_receptions_allowed'] = row2['total_receptions_allowed']
                playerSelection.loc[index, 'total_passing_yards_allowed'] = row2['total_passing_yards_allowed']
                playerSelection.loc[index, 'total_passing_plays_allowed'] = row2['total_passing_plays_allowed']

    # Prepare training data
    playerSelectionTrain = playerSelection[['receiving_yards','air_yards_x', 'air_yard_share', 'yards_after_catch',
                                            'completion?','total_yards_allowed', 'total_receptions_allowed',
                                            'total_passing_yards_allowed', 'total_passing_plays_allowed']]
    
    loo = LeaveOneOut()

    y_train= playerSelectionTrain.iloc[:,[0]].values#splitting data into train and test
    x_train = playerSelectionTrain.iloc[:,1:9].values

    reg = LinearRegression()

    # Perform K-Fold cross-validation and get the MSE for each fold
    cv_scores = cross_val_score(reg, x_train, y_train, cv=loo, scoring='neg_mean_squared_error')

    reg.fit(x_train, y_train)
    
    
    # Calculate averages for the next game prediction
    average_air_yards = playerSelectionTrain['air_yards_x'].median()
    average_air_yard_share = playerSelectionTrain['air_yard_share'].median()
    average_yards_after_catch = playerSelectionTrain['yards_after_catch'].median()
    average_completions = playerSelectionTrain['completion?'].median()
    average_yardsAllowed = df_defSelection['total_yards_allowed'].median()
    average_receptionsallowed = df_defSelection['total_receptions_allowed'].median()
    average_passingyardsallowed = df_defSelection['total_passing_yards_allowed'].median()
    average_passingplaysallowed = df_defSelection['total_passing_plays_allowed'].median()
    
    #print(average_yards_after_catch)
    #print(average_passingplaysallowed)

    next_game_data = [[average_air_yards, average_air_yard_share, average_yards_after_catch, average_completions, 
                       average_yardsAllowed, average_receptionsallowed, average_passingyardsallowed, 
                       average_passingplaysallowed]]
    
    predicted_yards = reg.predict(next_game_data)
    predicted_yards = predicted_yards.tolist()
 

    return jsonify({
        'player': playerInput,
        'defense': defenseInput,
        'predicted_yards': predicted_yards[0],
        'w': average_air_yards,
        'c':average_completions,
        'def':average_receptionsallowed
    })

if __name__ == '__main__':
    app.run(debug=True)
