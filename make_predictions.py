import pandas as pd
import numpy as np

from predictors import DummyPredictor, LightGBMPredictor, SklearnPredictor

from sklearn.metrics import f1_score


feats_csv = [
    ('years_ncaa', 'features/team_exp.csv'),
    ('coach', 'features/coaches.csv'),
    ('wins', 'features/record.csv'),
    ('seed', 'features/seeds.csv')
]
predictor = LightGBMPredictor(['coach_x', 'coach_y'])
# predictor = SklearnPredictor()
games_df = pd.read_csv('./features/games.csv')
feats_df = games_df
for feature, csv in feats_csv:
    df = pd.read_csv(csv)
    feats_df = pd.merge(feats_df, df, left_on=['year', 'home_team'], right_on=['year', 'team'])
    feats_df = feats_df.drop(['team'], axis=1)
    feats_df = pd.merge(feats_df, df, left_on=['year', 'away_team'], right_on=['year', 'team'])
    feats_df = feats_df.drop(['team'], axis=1)
feats_df = feats_df.drop(['year', 'home_team', 'away_team'], axis=1)
features = feats_df.drop('result', axis=1)
targets = feats_df.result.values.astype(np.uint8)
predictor.train(features, targets)


submission = pd.read_csv('/home/lsm/projects/kaggle/ncaa/data/MSampleSubmissionStage1_2020.csv')
test_games = pd.DataFrame()
test_games['year'] = submission.ID.apply(lambda x: x.split('_')[0])
test_games['home_team'] = submission.ID.apply(lambda x: x.split('_')[1])
test_games['away_team'] = submission.ID.apply(lambda x: x.split('_')[2])
test_games = test_games.reset_index()
for col in test_games:
    test_games[col] = test_games[col].apply(int)
feats_df_test = test_games
for feature, csv in feats_csv:
    df = pd.read_csv(csv)
    feats_df_test = pd.merge(feats_df_test, df, left_on=['year', 'home_team'], right_on=['year', 'team'])
    feats_df_test = feats_df_test.drop(['team'], axis=1)
    feats_df_test = pd.merge(feats_df_test, df, left_on=['year', 'away_team'], right_on=['year', 'team'])
    feats_df_test = feats_df_test.drop(['team'], axis=1)
    feats_df_test = feats_df_test.sort_values(by='index')
feats_df_test = feats_df_test.drop(['index'], axis=1)
feats_df_test = feats_df_test.drop(['year', 'home_team', 'away_team'], axis=1)
submission.Pred = predictor(feats_df_test)
submission.to_csv('./submission.csv', index=False)
pass