import streamlit as st
from recommender import load_data, create_matrix, train_similarity, item_based_recommend, user_based_recommend

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

st.title("🎬 Movie Recommender System")
st.markdown("**Collaborative Filtering (Item-Based & User-Based)**")

@st.cache_data
def prepare():
    ratings = load_data()
    matrix = create_matrix(ratings)
    item_sim_df, user_sim_df = train_similarity(matrix)
    return ratings, matrix, item_sim_df, user_sim_df

ratings, user_item_matrix, item_sim_df, user_sim_df = prepare()

user_ids = sorted(user_item_matrix.index.tolist())
user_id = st.selectbox("Select User ID", user_ids)

if st.button("Recommend Movies"):
    st.subheader("🎯 Item-Based Recommendations")
    item_recs = item_based_recommend(user_id, user_item_matrix, item_sim_df)
    for movie, score in item_recs:
        st.markdown(f"✅ {movie} — **Score:** {score:.2f}")

    st.subheader("👥 User-Based Recommendations")
    user_recs = user_based_recommend(user_id, user_item_matrix, user_sim_df)
    for movie, score in user_recs:
        st.markdown(f"🎬 {movie} — **Score:** {score:.2f}")
