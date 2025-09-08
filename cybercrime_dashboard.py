"""
Cybercrime Dashboard Application
--------------------------------

This Streamlit app visualizes trends in cybercrime complaints for U.S. states
across the years 2021–2024.  The underlying dataset was derived from the
``Top 10 States By Number Of Cybercrime Complaints And By Losses, 2024`` table
published by the Insurance Information Institute, which sources its figures
from the Internet Crime Complaint Center (IC3).  In addition to the 2024
complaint counts and losses listed in the table, the dataset includes
estimated complaint counts for previous years, calculated by applying a
5 percent annual decrease to each state's 2024 value.  For states not
appearing in the top‑ten list, complaint counts are set to zero and loss
figures are left as missing values.  This approach yields a complete table
containing every U.S. state and their complaint counts for four years.  When
building your own dashboard you should replace these estimates with actual
historic figures if they become available.

The dashboard offers the following features:

* A choropleth map showing complaint counts for the selected year across
  the United States.
* A heatmap comparing complaint counts over time for all states.
* Metric cards highlighting the state with the largest year‑over‑year
  increase (inbound) and decrease (outbound) in complaints.
* Donut charts indicating the percentage of states experiencing a
  significant increase (>5 000 complaints) or decrease (<−5 000 complaints)
  relative to the previous year.
* A sortable table listing states and their complaint counts for the
  selected year, accompanied by a progress bar for easy comparison.

To deploy this dashboard you will need to host both this Python file and
the ``cybercrime_top10.csv`` dataset (located in the same directory) in
your GitHub repository and point your Streamlit deployment to this file.

"""

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def format_number(num: float) -> str:
    """Return a human‑readable string for large numbers.

    Numbers larger than one million are converted to a string with an ``M``
    suffix and one decimal place (e.g. ``1500000`` → ``1.5 M``).  Numbers
    between one thousand and one million use a ``K`` suffix.  Smaller
    integers are returned unchanged.

    """
    try:
        n = float(num)
    except (ValueError, TypeError):
        return str(num)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return f"{int(n):,}"


def make_donut(percent: float, label: str, colour: str) -> alt.Chart:
    """Create a simple donut chart showing a percentage.

    ``percent`` should be between 0 and 100.  ``label`` is the label for
    the coloured portion of the chart.  ``colour`` selects one of four
    predefined colour schemes: ``'blue'``, ``'green'``, ``'orange'`` or
    ``'red'``.  The colour pairs were chosen to contrast against the dark
    Altair theme used throughout the dashboard.
    """
    # Define colour palettes for each scheme
    palette = {
        "blue": ["#29b5e8", "#155F7A"],
        "green": ["#27AE60", "#12783D"],
        "orange": ["#F39C12", "#875A12"],
        "red": ["#E74C3C", "#781F16"],
    }.get(colour, ["#29b5e8", "#155F7A"])

    # Data for the foreground and background arcs
    source = pd.DataFrame({"topic": [label, ""], "value": [percent, 100 - percent]})
    source_bg = pd.DataFrame({"topic": [label, ""], "value": [100, 0]})

    # Background ring (full 100 %)
    plot_bg = (
        alt.Chart(source_bg)
        .mark_arc(innerRadius=45, cornerRadius=20)
        .encode(
            theta="value",
            color=alt.Color(
                "topic:N",
                scale=alt.Scale(domain=[label, ""], range=palette),
                legend=None,
            ),
        )
        .properties(width=130, height=130)
    )

    # Foreground ring (highlighted portion)
    plot_fg = (
        alt.Chart(source)
        .mark_arc(innerRadius=45, cornerRadius=25)
        .encode(
            theta="value",
            color=alt.Color(
                "topic:N",
                scale=alt.Scale(domain=[label, ""], range=palette),
                legend=None,
            ),
        )
        .properties(width=130, height=130)
    )

    # Central text showing the percentage
    text = (
        plot_fg
        .mark_text(
            align="center",
            color=palette[0],
            fontSize=32,
            fontWeight=700,
            fontStyle="italic",
        )
        .encode(text=alt.value(f"{int(round(percent))} %"))
    )

    return plot_bg + plot_fg + text


def load_dataset() -> pd.DataFrame:
    """Load the cybercrime dataset and return a DataFrame.

    The CSV file ``cybercrime_top10.csv`` must reside in the same directory
    as this script.  It contains columns for state name, state abbreviation
    and complaint counts for 2021–2024.  If additional years or metrics are
    added in the future, the code below will need to be updated accordingly.
    """
    df = pd.read_csv("cybercrime_top10.csv")
    return df


def build_full_data(top_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with every U.S. state and complaint counts.

    ``top_df`` holds complaint counts for the top ten states.  This function
    combines it with a list of all U.S. state names and two‑letter postal
    codes.  States not present in ``top_df`` are assigned zero complaints
    and missing loss values.  The resulting DataFrame has columns:

        state, state_code, complaints_2024, complaints_2023,
        complaints_2022, complaints_2021, losses_2024_million

    where complaint columns are integers and loss values may be NaN.
    """
    # List of all states and their postal abbreviations
    us_states = [
        {"name": "Alabama", "code": "AL"},
        {"name": "Alaska", "code": "AK"},
        {"name": "Arizona", "code": "AZ"},
        {"name": "Arkansas", "code": "AR"},
        {"name": "California", "code": "CA"},
        {"name": "Colorado", "code": "CO"},
        {"name": "Connecticut", "code": "CT"},
        {"name": "Delaware", "code": "DE"},
        {"name": "District of Columbia", "code": "DC"},
        {"name": "Florida", "code": "FL"},
        {"name": "Georgia", "code": "GA"},
        {"name": "Hawaii", "code": "HI"},
        {"name": "Idaho", "code": "ID"},
        {"name": "Illinois", "code": "IL"},
        {"name": "Indiana", "code": "IN"},
        {"name": "Iowa", "code": "IA"},
        {"name": "Kansas", "code": "KS"},
        {"name": "Kentucky", "code": "KY"},
        {"name": "Louisiana", "code": "LA"},
        {"name": "Maine", "code": "ME"},
        {"name": "Maryland", "code": "MD"},
        {"name": "Massachusetts", "code": "MA"},
        {"name": "Michigan", "code": "MI"},
        {"name": "Minnesota", "code": "MN"},
        {"name": "Mississippi", "code": "MS"},
        {"name": "Missouri", "code": "MO"},
        {"name": "Montana", "code": "MT"},
        {"name": "Nebraska", "code": "NE"},
        {"name": "Nevada", "code": "NV"},
        {"name": "New Hampshire", "code": "NH"},
        {"name": "New Jersey", "code": "NJ"},
        {"name": "New Mexico", "code": "NM"},
        {"name": "New York", "code": "NY"},
        {"name": "North Carolina", "code": "NC"},
        {"name": "North Dakota", "code": "ND"},
        {"name": "Ohio", "code": "OH"},
        {"name": "Oklahoma", "code": "OK"},
        {"name": "Oregon", "code": "OR"},
        {"name": "Pennsylvania", "code": "PA"},
        {"name": "Rhode Island", "code": "RI"},
        {"name": "South Carolina", "code": "SC"},
        {"name": "South Dakota", "code": "SD"},
        {"name": "Tennessee", "code": "TN"},
        {"name": "Texas", "code": "TX"},
        {"name": "Utah", "code": "UT"},
        {"name": "Vermont", "code": "VT"},
        {"name": "Virginia", "code": "VA"},
        {"name": "Washington", "code": "WA"},
        {"name": "West Virginia", "code": "WV"},
        {"name": "Wisconsin", "code": "WI"},
        {"name": "Wyoming", "code": "WY"},
    ]

    records: list[dict] = []
    for item in us_states:
        name = item["name"]
        code = item["code"]
        # Look up the row for this state in the top‑ten dataframe
        row = top_df[top_df["state"] == name]
        # Start building the record
        record: dict[str, object] = {
            "state": name,
            "state_code": code,
        }
        # For each year assign the actual value or zero
        for year in (2024, 2023, 2022, 2021):
            col = f"complaints_{year}"
            if not row.empty:
                record[col] = int(row.iloc[0][col])
            else:
                record[col] = 0
        # Set losses where available
        if not row.empty:
            record["losses_2024_million"] = row.iloc[0]["losses_2024_million"]
        else:
            record["losses_2024_million"] = np.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


def prepare_heatmap_data(full_df: pd.DataFrame) -> pd.DataFrame:
    """Transform the wide DataFrame into long form for Altair heatmap.

    The input has complaint counts across multiple year columns.  This
    function melts those columns into two columns: ``year`` and
    ``complaints``.  The ``year`` column is converted to an integer for
    proper sorting.
    """
    # Select only complaint columns and state
    complaint_cols = [c for c in full_df.columns if c.startswith("complaints_")]
    melted = full_df.melt(
        id_vars=["state"], value_vars=complaint_cols, var_name="year", value_name="complaints"
    )
    # Extract the numeric year from the column name
    melted["year"] = melted["year"].str.replace("complaints_", "").astype(int)
    return melted


def main() -> None:
    """Entry point for the Streamlit app."""
    # Configure the page and enable dark theme for Altair
    st.set_page_config(
        page_title="US Cybercrime Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    alt.themes.enable("dark")

    # Load and prepare data
    top_df = load_dataset()
    full_df = build_full_data(top_df)

    # Sidebar controls
    with st.sidebar:
        st.title("🛡️ US Cybercrime Dashboard")
        # Select year; list reversed so the most recent year appears first
        years = [2024, 2023, 2022, 2021]
        selected_year = st.selectbox("Select a year", years, index=0)
        # Colour themes for the choropleth
        color_theme_list = [
            "blues",
            "cividis",
            "greens",
            "inferno",
            "magma",
            "plasma",
            "reds",
            "rainbow",
            "turbo",
            "viridis",
        ]
        selected_theme = st.selectbox("Select a colour theme", color_theme_list)

    # Prepare selected year's data
    col_name = f"complaints_{selected_year}"
    df_selected_year = full_df[["state", "state_code", col_name, "losses_2024_million"]].copy()
    df_selected_year = df_selected_year.rename(columns={col_name: "complaints"})
    # Sort states by complaints descending
    df_selected_year_sorted = df_selected_year.sort_values(by="complaints", ascending=False)

    # Prepare heatmap data (once)
    heatmap_df = prepare_heatmap_data(full_df)

    # Compute year‑over‑year differences for metrics and migration percentages
    if selected_year > 2021:
        prev_col = f"complaints_{selected_year - 1}"
        df_diff = full_df[["state", "state_code", col_name, prev_col]].copy()
        df_diff["difference"] = df_diff[col_name] - df_diff[prev_col]
        # Determine top positive and negative changes
        top_gain = df_diff.loc[df_diff["difference"].idxmax()]
        top_loss = df_diff.loc[df_diff["difference"].idxmin()]
        # Calculate the proportion of states with significant migration
        inbound_states = (df_diff["difference"] > 5000).sum()
        outbound_states = (df_diff["difference"] < -5000).sum()
        total_states = len(df_diff)
        inbound_percent = inbound_states / total_states * 100
        outbound_percent = outbound_states / total_states * 100
    else:
        # First year has no previous year for comparison
        top_gain = None
        top_loss = None
        inbound_percent = 0
        outbound_percent = 0

    # Layout columns
    col = st.columns((1.5, 4.5, 2), gap="medium")

    # Column 1: Metrics and migration donut
    with col[0]:
        st.markdown("#### Gains/Losses")
        if top_gain is not None and top_loss is not None:
            # Display the state with the largest increase
            gain_state = top_gain["state"]
            gain_value = format_number(top_gain[col_name])
            gain_delta = format_number(top_gain["difference"])
            st.metric(label=gain_state, value=gain_value, delta=f"+{gain_delta}")

            # Display the state with the largest decrease
            loss_state = top_loss["state"]
            loss_value = format_number(top_loss[col_name])
            loss_delta = format_number(top_loss["difference"])
            st.metric(label=loss_state, value=loss_value, delta=loss_delta)
        else:
            # For 2021 (no previous year) display placeholder metrics
            st.metric(label="–", value="–", delta="")
            st.metric(label="–", value="–", delta="")

        st.markdown("#### States Migration")
        if selected_year > 2021:
            donut_inbound = make_donut(inbound_percent, "Inbound Migration", "green")
            donut_outbound = make_donut(outbound_percent, "Outbound Migration", "red")
        else:
            donut_inbound = make_donut(0, "Inbound Migration", "green")
            donut_outbound = make_donut(0, "Outbound Migration", "red")
        # Display the donut charts
        mig_cols = st.columns((0.2, 1, 0.2))
        with mig_cols[1]:
            st.write("Inbound")
            st.altair_chart(donut_inbound, use_container_width=True)
            st.write("Outbound")
            st.altair_chart(donut_outbound, use_container_width=True)

    # Column 2: Choropleth and heatmap
    with col[1]:
        st.markdown("#### Total Complaints by State")
        # Choropleth map using Plotly
        fig = px.choropleth(
            df_selected_year,
            locations="state_code",
            color="complaints",
            locationmode="USA-states",
            color_continuous_scale=selected_theme,
            range_color=(0, df_selected_year["complaints"].max()),
            scope="usa",
            labels={"complaints": "Complaints"},
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap showing complaints over years and states
        heatmap = (
            alt.Chart(heatmap_df)
            .mark_rect()
            .encode(
                y=alt.Y(
                    "year:O",
                    axis=alt.Axis(
                        title="Year",
                        titleFontSize=16,
                        titleFontWeight=600,
                        labelAngle=0,
                    ),
                ),
                x=alt.X(
                    "state:O",
                    sort=alt.SortField(field="complaints", op="sum", order="descending"),
                    axis=alt.Axis(title="State", titleFontSize=16, titleFontWeight=600),
                ),
                color=alt.Color(
                    "complaints:Q",
                    scale=alt.Scale(scheme=selected_theme),
                    legend=alt.Legend(title="Complaints"),
                ),
                stroke=alt.value("black"),
                strokeWidth=alt.value(0.25),
            )
            .properties(width=900, height=300)
            .configure_axis(labelFontSize=10, titleFontSize=12)
        )
        st.altair_chart(heatmap, use_container_width=True)

    # Column 3: Data table and description
    with col[2]:
        st.markdown("#### Top States by Complaints")
        st.dataframe(
            df_selected_year_sorted,
            column_order=("state", "complaints", "losses_2024_million"),
            hide_index=True,
            use_container_width=True,
            column_config={
                "state": st.column_config.TextColumn("State"),
                "complaints": st.column_config.ProgressColumn(
                    "Complaints",
                    format="{:,.0f}",
                    min_value=0,
                    max_value=int(df_selected_year_sorted["complaints"].max()),
                ),
                "losses_2024_million": st.column_config.NumberColumn(
                    "Losses 2024 (USD m)",
                    format="{:,.0f}",
                ),
            },
        )
        with st.expander("About", expanded=True):
            st.write(
                """
                **Data source:** The complaint counts and financial losses were
                compiled from the Insurance Information Institute's 2024 table of
                the top ten states by cybercrime complaints and losses, which
                relies on the Internet Crime Complaint Center for its figures【72692044984918†screenshot】.
                Complaint counts for earlier years were estimated by applying a
                5 % annual decrease to each state's 2024 value.

                **Gains/Losses:** These metrics display the states with the
                largest absolute increase and decrease in complaints when
                compared against the previous year.

                **States Migration:** Shows the percentage of states
                experiencing a net increase greater than 5 000 complaints or a
                net decrease greater than 5 000 complaints relative to the
                previous year.
                """
            )


if __name__ == "__main__":
    main()