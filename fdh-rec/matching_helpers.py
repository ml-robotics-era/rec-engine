"""
Contains functions that enforce base conditions, rank products by demographic and 
attribute similarity with users.
"""
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import datetime
from typing import Dict, List, Tuple
from utils import (
    test,
    RECOMMENDED,
    PURCHASED,
    ZERO,
    OTHER,
    NATURAL,
    SIMILARITY_BOOST,
    DEVICE,
    VIRTUAL,
    SUBTRACT_VIRTUAL,
    ADD_DEVICE,
    POPULARITY,
    SHRINK_CAC,
    DEMO_PRIORITY,
)


def filter_pids(
    uid: int,
    status: pd.DataFrame,
    user_features: pd.DataFrame,
    prod_features: pd.DataFrame,
    user_goals: pd.DataFrame,
    user_issues: pd.DataFrame,
    prod_goals: pd.DataFrame,
    prod_issues: pd.DataFrame,
) -> Tuple[pd.Series, bool]:
    """
    Filters for products that meet base conditions for a given user.

    Paramters:
      uid: User id
      Relevant DataFrames.

    Returns:
      eligible_pids: Series of pid's of products that satisfies conditions for user.
      wellness_flag: Boolean indicating whether the user is only recommended wellness
        products, due to the user not specifying goals or issues or the absence of other
        eligible products.
    This function will only return empty if not even wellness products are found for the user.
    """
    wellness_flag = False
    past_prods = status[status.uid == uid]

    # Constraint 1 is user cannot have already been recommended or bought this product.
    used_pids = set(
        past_prods[
            (past_prods.status == RECOMMENDED) | (past_prods.status == PURCHASED)
        ].pid
    )

    # Constraint 2 is sex must match.
    u = user_features.loc[uid].to_numpy()
    sex, age, weight, income_l, income_r, price_ceil, tier = u
    wrong_sex_idx = set()
    if sex != OTHER:  # If user's sex is "other", all products satisfy this condition.
        wrong_sex_idx = set(
            prod_features[prod_features.sex.apply(lambda x: sex not in x)].index
        )
    used_pids |= wrong_sex_idx

    if (user_goals.loc[uid] == ZERO).all() and (user_issues.loc[uid] == ZERO).all():
        eligible_pids = prod_goals[
            (~prod_goals.index.isin(used_pids)) & (prod_goals.wellness != ZERO)
        ].index
        wellness_flag = True
    else:
        # Constraint 3 is if user has goals or issues, the algorithm must match at
        # least one goal or issue.
        goal_match_vec = (
            prod_goals[~prod_goals.index.isin(used_pids)]
            .apply(lambda x: (x @ user_goals.loc[uid]) > ZERO, axis=1)
            .astype("int")
        )
        issue_match_vec = (
            prod_issues[~prod_issues.index.isin(used_pids)]
            .apply(lambda x: (x @ user_issues.loc[uid]) > ZERO, axis=1)
            .astype("int")
        )
        total_vec = goal_match_vec + issue_match_vec
        eligible_pids = total_vec[total_vec > ZERO].index
        if (
            eligible_pids.empty
        ):  # If no more relevant products are found for this user, only recommend
            # products with wellness goals.
            eligible_pids = prod_goals[
                (~prod_goals.index.isin(used_pids)) & (prod_goals.wellness != ZERO)
            ].index
            wellness_flag = True
    return (
        eligible_pids,
        wellness_flag,
    )


def demo_score(
    uid: int,
    eligible_features: pd.DataFrame,
    user_features: pd.DataFrame,
    demo_params: Dict,
) -> np.array:
    """
    Calculates the demographic matching score for a single user and all products that
    already satisfy base conditions.

    Paramters:
      uid: User id.
      demo_params: Dictionary of importance weights on different demographic attributes.
      Relevant DataFrames.

    Returns: Numpy array of demographics score per product for this user.
    """
    u = user_features.loc[uid].to_numpy()
    _, age, weight, income_l, income_r, price_ceil, tier = u

    # Logical_and checks if the user's demographic attributes are within range
    # of products' attributes.
    age_vec = np.logical_and(
        eligible_features.age_l.apply(
            lambda x: x <= age
        ),  # Vector of whether eligible products' age lower bounds are less than user age.
        eligible_features.age_r.apply(
            lambda x: x >= age
        ),  # Vector of whether eligible products' age upper bounds are greater than user age.
    ).astype("int")

    weight_vec = np.logical_and(
        eligible_features.weight_l.apply(lambda x: x <= weight),
        eligible_features.weight_r.apply(lambda x: x >= weight),
    ).astype("int")

    income_vec = np.logical_and(
        eligible_features.income_l.apply(lambda x: x <= income_l),
        eligible_features.income_r.apply(lambda x: x >= income_r),
    ).astype("int")

    # POST-BETA: If user sets a maximum acceptable price through surveys, this vector
    # represents whether product prices are below that.
    net_pay = eligible_features.price - tier * eligible_features.cac
    price_vec = (net_pay <= price_ceil).astype("int")

    vec_list = [age_vec, weight_vec, income_vec, price_vec]
    weights = list(demo_params.values())
    for i in range(len(weights)):
        vec_list[i] = vec_list[i] * weights[i]
    scores = np.vstack(vec_list)
    # Dividing by the number of attributes balances out the score.
    return np.sum(scores, axis=0) / len(vec_list)


def final_match(
    k: int,
    status: pd.DataFrame,
    user_features: pd.DataFrame,
    prod_features: pd.DataFrame,
    user_goals: pd.DataFrame,
    user_issues: pd.DataFrame,
    user_subcats: pd.DataFrame,
    prod_goals: pd.DataFrame,
    prod_issues: pd.DataFrame,
    prod_subcats: pd.DataFrame,
    prod_cos_frame: pd.DataFrame,
    user_cos_frame: pd.DataFrame,
    global_params: Dict,
    prod_params: Dict,
    demo_params: Dict,
) -> List[Dict]:
    """
    Computes cosine similarity between all users and products and adds adjustments.

    Paramaters:
      global_params: Dictionary of miscellaneous weights.
      prod_params: Dictionary of importance weights on different product attributes.
      demo_params: Dictionary of importance weights on different demographic attributes.
      relevant DataFrames.

    Returns: Python list of dictionaries following the json output format.
    """
    out = []
    for j in user_features.index:

        pids, wellness_flag = filter_pids(
            j,
            status,
            user_features,
            prod_features,
            user_goals,
            user_issues,
            prod_goals,
            prod_issues,
        )
        if pids.empty:  # Not even wellness products are found.
            out.append(
                {
                    "uid": j,
                    "pid": None,
                    "score": None,
                    "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                }
            )
            continue
        eligible_prods = prod_cos_frame.loc[pids]
        eligible_features = prod_features.loc[pids]
        eligible_features[NATURAL] = range(len(eligible_features))

        similarity = (
            cosine_similarity(
                user_cos_frame.loc[j].to_numpy().reshape(1, -1), eligible_prods
            )[
                0
            ]  # This is a vector with natural-number indexing.
            * global_params[SIMILARITY_BOOST]
        )

        # Boost products that are devices and discourage virtual products.
        device_idx = eligible_features[
            eligible_features.overcat.apply(lambda x: DEVICE in x)
        ].natural.to_numpy()
        virtual_idx = eligible_features[
            eligible_features.overcat.apply(lambda x: VIRTUAL in x)
        ].natural.to_numpy()
        similarity[virtual_idx] -= global_params[SUBTRACT_VIRTUAL]
        similarity[device_idx] += global_params[ADD_DEVICE]

        # Boosts products that have lots of purchases or review requests out of
        # the total times they are recommended.
        popularity = np.arctan(
            (eligible_features.n_request + eligible_features.n_purchase)
            / eligible_features.n_recommend
        ).fillna(0) / (
            np.pi / 2
        )  # Arctan maps range [0, inf] to [0, 1].

        similarity += popularity * prod_params[POPULARITY]
        similarity += eligible_features.cac / global_params[SHRINK_CAC]
        similarity += (
            demo_score(j, eligible_features, user_features, demo_params)
            * global_params[DEMO_PRIORITY]
        )

        enum = list(enumerate(similarity))
        best_scores = sorted(enum, key=lambda x: x[1], reverse=True)[
            :k
        ]  # Reverse=True matches from top-down.
        indices = [i[0] for i in best_scores]
        vals = [i[1] for i in best_scores]
        recs = pids[indices]  # Convert natural indices back to pids.

        try:
            test(
                j,
                pids[indices],
                wellness_flag,
                prod_goals,
                prod_issues,
                prod_subcats,
                status,
                user_goals,
                user_issues,
                user_subcats,
                user_features,
                prod_features,
            )
        except AssertionError as e:
            # In case of a repeat recommendation, sex mismatch, or failure to find a
            # match when one exists, throw error and continue loop.
            print(e)  # TODO: Change from print to log statements.

        # If json: 
        # out.append(
        #     {
        #         "uid": int(j),
        #         "pid": int(recs[0]),
        #         "score": vals[0],
        #         "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        #     }
        # )
        out.append((int(j), int(recs[0]),vals[0], datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    return out
