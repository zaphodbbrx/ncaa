import re
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

def team_features(csv_path):
    results_df = pd.read_csv(csv_path)


def prepare_targets(csv_path):
    results_df = pd.read_csv(csv_path)

    hteam1 = results_df[results_df.WLoc == 'H']['WTeamID']
    hteam2 = results_df[results_df.WLoc == 'A']['LTeamID']
    ateam1 = results_df[results_df.WLoc == 'H']['LTeamID']
    ateam2 = results_df[results_df.WLoc == 'A']['WTeamID']
    results_df.loc[hteam1.index.tolist(), 'home_team'] = hteam1
    results_df.loc[hteam2.index.tolist(), 'home_team'] = hteam2
    results_df.loc[ateam1.index.tolist(), 'away_team'] = ateam1
    results_df.loc[ateam2.index.tolist(), 'away_team'] = ateam2
    res1 = results_df[results_df.WLoc == 'H']
    res2 = results_df[results_df.WLoc == 'A']
    results_df.loc[res1.index.tolist(), 'result'] = 1.0
    results_df.loc[res2.index.tolist(), 'result'] = 0.0
    neutral_games = results_df[results_df.WLoc == 'N'].sample(frac=1.0)
    ngames1 = neutral_games.shape[0] // 2
    neutral_games1 = neutral_games.iloc[:ngames1]
    neutral_games2 = neutral_games.iloc[ngames1:]
    hteam1 = neutral_games1.WTeamID
    ateam1 = neutral_games1.LTeamID
    hteam2 = neutral_games2.LTeamID
    ateam2 = neutral_games2.WTeamID
    results_df.loc[hteam1.index.tolist(), 'home_team'] = hteam1
    results_df.loc[hteam2.index.tolist(), 'home_team'] = hteam2
    results_df.loc[ateam1.index.tolist(), 'away_team'] = ateam1
    results_df.loc[hteam2.index.tolist(), 'away_team'] = ateam2

    results_df.loc[hteam1.index.tolist(), 'result'] = 0.0
    results_df.loc[ateam1.index.tolist(), 'result'] = 1.0
    results_df.loc[hteam2.index.tolist(), 'result'] = 1.0
    results_df.loc[ateam2.index.tolist(), 'result'] = 0.0
    results_df = results_df.dropna()
    results_df['year'] = results_df.Season
    results_df['home_team'] = results_df['home_team'].apply(int)
    results_df['away_team'] = results_df['away_team'].apply(int)
    return results_df[['year', 'home_team', 'away_team', 'result']]


def team_features():
    teams_df = pd.read_csv('data/MDataFiles_Stage1/MTeams.csv')
    games_df = pd.read_csv('features/games.csv')
    year_team_df = pd.DataFrame([yt for yt, df in games_df.groupby(['year', 'home_team'])], columns=['year', 'team'])
    feats_df = pd.merge(year_team_df, teams_df, left_on=['team'], right_on=['TeamID'])
    feats_df['years_ncaa'] = feats_df['year'] - feats_df['FirstD1Season']
    return feats_df[['year', 'team', 'years_ncaa']]


def team_record_features():
    results_df = pd.read_csv('data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
    wins_df = results_df.groupby(['Season', 'WTeamID']).agg({'NumOT': len}).reset_index()
    wins_df['wins'] = wins_df['NumOT']
    wins_df = wins_df.rename(columns={'Season': 'year', 'WTeamID': 'team'})
    loss_df = results_df.groupby(['Season', 'LTeamID']).agg({'NumOT': len}).reset_index()
    loss_df['losses'] = loss_df['NumOT']
    loss_df = loss_df.rename(columns={'Season': 'year', 'LTeamID': 'team'})
    res_df = wins_df
    res_df['losses'] = loss_df['losses']
    res_df['win_pct'] = res_df['wins'] / (res_df['wins'] + res_df['losses'])
    return res_df[['year', 'team', 'wins', 'losses']]


def coaches_features():
    coaches_df = pd.read_csv('data/MDataFiles_Stage1/MTeamCoaches.csv')
    coaches_df = coaches_df.rename(columns={'Season': 'year', 'TeamID': 'team', 'CoachName': 'coach'})
    all_coaches = coaches_df.coach.unique().tolist()
    coaches_dict = dict(zip(all_coaches, range(len(all_coaches))))
    coaches_df['coach'] = coaches_df['coach'].map(coaches_dict)
    coaches_df = coaches_df.groupby(['year', 'team']).agg({'coach': max}).reset_index()
    return coaches_df[['year', 'team', 'coach']]


def seed_data():
    seed_df = pd.read_csv('/home/lsm/projects/kaggle/ncaa/data/MDataFiles_Stage1/MNCAATourneySeeds.csv')
    game_exp_df = pd.read_csv('/home/lsm/projects/kaggle/ncaa/features/team_exp.csv')
    seed_df['seed'] = seed_df['Seed'].apply(lambda x: int(re.sub('[a-zA-Z]', '', x)))
    seed_df['year'] = seed_df['Season']
    seed_df['team'] = seed_df['TeamID']
    seed_df = pd.merge(game_exp_df, seed_df, left_on=['year', 'team'], right_on=['year', 'team'], how='inner')
    return seed_df[['year', 'team', 'seed']].fillna(32)


if __name__ == '__main__':
    results_df = prepare_targets('/home/lsm/projects/kaggle/ncaa/data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
    results_df.to_csv('features/games.csv', index=False)
    # feats_df = team_features()
    # feats_df.to_csv('features/team_exp.csv', index=False)
    # record_df = team_record_features()
    # record_df.to_csv('features/record.csv', index=False)
    # coaches_df = coaches_features()
    # coaches_df.to_csv('features/coaches.csv', index=False)
    seed_df = seed_data()
    seed_df.to_csv('features/seeds.csv', index=False)
    pass