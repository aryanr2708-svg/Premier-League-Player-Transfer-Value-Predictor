import pandas as pd
import streamlit as st

from config import DATA_DIR


st.set_page_config(
    page_title="Premier League Transfer Value Predictor",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Premier League Player Transfer Value Predictor")

df = pd.read_csv(
    f"{DATA_DIR}/predicted_values.csv",
    encoding="utf-8-sig",
)

df = df.sort_values(
    "predicted_market_value_eur",
    ascending=False,
)


def format_value(value):
    if value >= 1_000_000:
        return f"€{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"€{value / 1_000:.0f}K"
    else:
        return f"€{value:.0f}"


for _, player in df.head(20).iterrows():

    photo_col, info_col, value_col = st.columns([1, 4, 2])

    with photo_col:
        image_url = player.get("image_url")

        if pd.notna(image_url) and image_url:
            st.image(image_url, width=100)

    with info_col:
        st.subheader(player["player_name"])
        st.write(
            f"**{player['team']}** · "
            f"{player['position_group']} · "
            f"Age {player['age']}"
        )

        st.write(
            f"Goals: **{player['goals']}**  |  "
            f"Assists: **{player['assists']}**"
        )

    with value_col:
        st.metric(
            "Predicted Transfer Value",
            format_value(player["predicted_market_value_eur"]),
        )

    st.divider()