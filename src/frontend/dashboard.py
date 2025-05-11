import streamlit as st
import pandas as pd
import pickle
from collections import defaultdict
from surprise import Dataset, Reader

# Load data
df = pd.read_csv("../../data/reviews_stratified_sampled.csv")

# Model options
model_options = {
    "User-based CF": "../../model/user_based_collaborative_filtering_baseline.pkl",
    "Item-based CF": "../../model/item_based_collaborative_filtering_baseline.pkl"
}

# Select model
selected_model_name = st.selectbox("Select a recommendation model:", list(model_options.keys()))

# Load the selected model
with open(model_options[selected_model_name], "rb") as f:
    model = pickle.load(f)

# Dropdowns for user and beer selection
target_user = 'username'
target_beer = 'beer_id'

unique_users = df[target_user].unique()
unique_beers = df[target_beer].unique()

selected_user = st.selectbox(f"Select a {target_user}:", unique_users)
selected_beer = st.selectbox(f"Select a {target_beer}:", unique_beers)

left_column, right_column = st.columns(2)

# Predict single rating
if left_column.button("Predict rating"):
    prediction = model.predict(selected_user, selected_beer)
    st.write(f"Predicted rating by {selected_model_name} for user `{selected_user}` on beer `{selected_beer}`: **{prediction.est:.2f}**")

# Top-N function
def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

# Show top-N recommendations
if right_column.button("Show Top 10 Recommendations"):
    user_rated_beers = df[df[target_user] == selected_user][target_beer].unique()
    beers_to_predict = [iid for iid in unique_beers if iid not in user_rated_beers]

    prediction_input = [(selected_user, iid, 0) for iid in beers_to_predict]
    predictions = model.test(prediction_input)

    top_n = get_top_n(predictions, n=10)
    user_recommendations = top_n.get(selected_user, [])

    st.write(f"### Top 10 Recommended Beers for {selected_user}:")
    for beer_id, est_rating in user_recommendations:
        st.write(f"Beer ID: `{beer_id}` — Estimated Rating: **{est_rating:.2f}**")
