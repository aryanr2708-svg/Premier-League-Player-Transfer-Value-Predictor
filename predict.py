import joblib
import matplotlib.pyplot as plt
import pandas as pd

from config import FEATURES_FILE, MODEL_FILE, DATA_DIR
from train_model import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def predict_all() -> pd.DataFrame:
    pipeline = joblib.load(MODEL_FILE)

    df = pd.read_csv(
        FEATURES_FILE,
        encoding="utf-8-sig"
    )


    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
        )


    X = df[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ]

    df["predicted_market_value_eur"] = pipeline.predict(X)

    if "image_url" not in df.columns:
        df["image_url"] = ""


    df["image_url"] = (
        df["image_url"]
        .fillna("")
        .astype(str)
        .replace(
            ["nan", "NaN", "None", "null"],
            "",
            regex=False
        )
        .str.strip()
    )

    out_path = f"{DATA_DIR}/predicted_values.csv"

    cols = [
        "player_name",
        "team",
        "position_group",
        "age",
        "goals",
        "assists",
        "image_url",
        "predicted_market_value_eur",
    ]

    ranked = (
        df[cols]
        .sort_values(
            "predicted_market_value_eur",
            ascending=False
        )
    )

    # Save predictions
    ranked.to_csv(
        out_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Saved predictions for {len(df)} players to {out_path}"
    )


    def _abbreviate(v: float) -> str:
        if v >= 1_000_000:
            return f"€{v / 1_000_000:.1f}M"

        if v >= 1_000:
            return f"€{v / 1_000:.0f}K"

        return f"€{v:.0f}"

    top_n = 20

    print(
        f"\nTop {top_n} predicted market values:\n"
    )

    display = ranked.head(top_n).copy()

    display["predicted_market_value_eur"] = (
        display["predicted_market_value_eur"]
        .map(_abbreviate)
    )

    print(
        display.to_string(index=False)
    )

    _show_chart(
        ranked.head(top_n)
    )

    _generate_html_report(
        ranked.head(top_n)
    )

    return df


def _generate_html_report(
    top_df: pd.DataFrame,
    out_path: str = f"{DATA_DIR}/report.html"
) -> None:

    rows = ""

    for _, row in top_df.iterrows():

        image_url = row["image_url"]

        if pd.isna(image_url):
            image_url = ""

        image_url = str(image_url).strip()

        if image_url and image_url.lower() not in {
            "nan",
            "none",
            "null"
        }:

            photo_html = (
                f'<img src="{image_url}" '
                f'width="60" height="60" '
                f'style="border-radius:50%;'
                f'object-fit:cover;" '
                f'onerror="this.style.display=\'none\';">'
            )

        else:

            photo_html = (
                '<div style="width:60px;height:60px;'
                'border-radius:50%;'
                'background:#ddd;'
                'display:flex;'
                'align-items:center;'
                'justify-content:center;'
                'font-size:24px;">'
                '📷'
                '</div>'
            )

        rows += f"""
        <tr>
            <td>
                {photo_html}
            </td>
            <td>{row['player_name']}</td>
            <td>{row['team']}</td>
            <td>{row['age']}</td>
            <td>
                €{row['predicted_market_value_eur'] / 1_000_000:.1f}M
            </td>
        </tr>
        """

    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: sans-serif;
            }}

            table {{
                border-collapse: collapse;
                width: 100%;
            }}

            th, td {{
                padding: 8px 12px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }}

            th {{
                background: #222;
                color: white;
            }}
        </style>
    </head>

    <body>

        <h2>Top Predicted Player Values</h2>

        <table>

            <tr>
                <th>Photo</th>
                <th>Player</th>
                <th>Team</th>
                <th>Age</th>
                <th>Predicted Value</th>
            </tr>

            {rows}

        </table>

    </body>
    </html>
    """

    with open(
        out_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)

    print(
        f"Report with player photos saved to "
        f"{out_path} — open it in your browser."
    )


def _show_chart(
    top_df: pd.DataFrame
) -> None:

    """Pops up a horizontal bar chart of the top predicted players."""

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    labels = (
        top_df["player_name"]
        + " ("
        + top_df["team"]
        + ")"
    )

    values_m = (
        top_df["predicted_market_value_eur"]
        / 1_000_000
    )

    ax.barh(
        labels[::-1],
        values_m[::-1],
        color="#3d5afe"
    )

    ax.set_xlabel(
        "Predicted market value (€M)"
    )

    ax.set_title(
        "Top predicted Premier League player market values"
    )

    for i, v in enumerate(
        values_m[::-1]
    ):

        ax.text(
            v,
            i,
            f" €{v:.1f}M",
            va="center"
        )

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    predict_all()