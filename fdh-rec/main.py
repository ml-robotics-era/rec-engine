"""
Loads in DataFrames from the data folder, passes them through preprocessing, 
and calls the matching algorithm.
Returns: A json array of dictionaries, with null values if no valid recommendations
are found for a user.
"""
# is this wrapper going to be main now? 
import json
from matching_helpers import final_match
from data import preprocess, fill_indices
from utils import TEST_PRINT_K
import numpy as np
import pandas as pd
import postgres

np.random.seed(0)  # This seed is shared across the entire prorgam.


def main():
    # DEPLOY: load from PostgreSQL. TODO: fix read sql
    conn = postgres.connectionFrom(ENV.dbConnectionString)
    prod_features = pd.read_sql("select * prod_features_view", conn)
    prod_goals= pd.read_sql('SELECT * FROM prod_goals_view', conn)
    prod_issues= pd.read_sql('SELECT * FROM prod_issues_view', conn)
    prod_subcats= pd.read_sql('SELECT * FROM prod_subcats_view', conn)
    user_features= pd.read_sql('SELECT * FROM user_features_view', conn)
    user_goals= pd.read_sql('SELECT * FROM user_goals_view', conn)
    user_issues= pd.read_sql('SELECT * FROM user_issues_view', conn)
    user_subcats= pd.read_sql('SELECT * FROM user_subcats_view', conn)
    demo_params = pd.read_sql('SELECT * FROM demo_params', conn)
    global_params = pd.read_sql('SELECT * FROM global_params', conn)
    goal_ranking = pd.read_sql('SELECT * FROM goal_ranking', conn)
    issues_ranking = pd.read_sql('SELECT * FROM issues_ranking', conn)
    prod_params = pd.read_sql('SELECT * FROM prod_params', conn)
    status = pd.read_sql('SELECT * FROM status', conn)

    pd.set_option('display.expand_frame_repr', False)
    
    # EXPERIMENT: load from csv.
    # demo_params = pd.read_csv("data/demo_params.csv").to_dict("records")[0]
    # global_params = pd.read_csv("data/global_params.csv").to_dict("records")[0]
    # goal_ranking = pd.read_csv("data/goal_ranking.csv").to_dict("records")[0]
    # issues_ranking = pd.read_csv("data/issues_ranking.csv").to_dict("records")[0]
    # prod_params = pd.read_csv("data/prod_params.csv").to_dict("records")[0]
    # status = pd.read_csv("data/status.csv")
    # prod_features = pd.read_csv("data/prod_features.csv", index_col=0)
    # goal_ranking = pd.read_csv("goal_ranking.csv").set_index('goal_name').to_dict()['ranking']
    # issues_ranking = pd.read_csv("issues_ranking.csv").set_index('issue_name').to_dict()['ranking']
    # prod_subcats = pd.read_csv("data/prod_subcats.csv", index_col=0)
    # user_features = pd.read_csv("data/user_features.csv", index_col=0)
    # user_goals = pd.read_csv("data/user_goals.csv", index_col=0)
    # user_issues = pd.read_csv("data/user_issues.csv", index_col=0)
    # user_subcats = pd.read_csv("data/user_subcats.csv", index_col=0)

    n_goals = len(prod_goals.columns)
    n_issues = len(prod_issues.columns)

    # Remove the call below upon deployment.
    (
        prod_features,
        prod_goals,
        prod_issues,
        prod_subcats,
        user_features,
        user_goals,
        user_issues,
        user_subcats,
    ) = fill_indices(
        prod_features,
        prod_goals,
        prod_issues,
        prod_subcats,
        user_features,
        user_goals,
        user_issues,
        user_subcats,
    )

    (
        user_subcats,
        prod_features,
        prod_goals,
        prod_issues,
        user_cos_frame,
        prod_cos_frame,
    ) = preprocess(
        prod_features,
        prod_goals,
        prod_issues,
        prod_subcats,
        user_goals,
        user_issues,
        user_subcats,
        global_params,
        n_goals,
        n_issues,
        goal_ranking,
        issues_ranking,
    )

    out = final_match(
        TEST_PRINT_K,
        status,
        user_features,
        prod_features,
        user_goals,
        user_issues,
        user_subcats,
        prod_goals,
        prod_issues,
        prod_subcats,
        prod_cos_frame,
        user_cos_frame,
        global_params,
        prod_params,
        demo_params,
    )

    conn.startTransaction()
    values = ', '.join(map(str, out))
    sql = "INSERT INTO recommendations VALUES {}".format(values)
    conn.query(sql)
    conn.close()
    #EXPERIMENT:
    #json.dumps(out, indent=1)


if __name__ == "__main__":
    main()
