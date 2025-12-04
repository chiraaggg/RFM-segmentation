import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import altair as alt

st.set_page_config(page_title="🧠 RFM + KMeans Segmentation", layout="wide")
st.title("🧠 RFM Segmentation Dashboard (with KMeans Option)")

st.markdown("Upload a single `orders.csv` file with the following columns:")
st.code(
    "user_id, created_at, total, sub_total, discount, coupon_id, "
    "payment_method, actual_qty, user_created_at, name, phone"
)

uploaded_file = st.file_uploader("📤 Upload orders.csv", type=["csv"])

# =================================================================================================

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["user_created_at"] = pd.to_datetime(df["user_created_at"])
    today = pd.to_datetime("today")

    # -------------------- RFM --------------------
    rfm = df.groupby("user_id").agg(
        last_order_date=("created_at", "max"),
        frequency=("user_id", "count"),
        monetary=("total", "sum"),
        user_created_at=("user_created_at", "min"),
        name=("name", "first"),
        phone=("phone", "first"),
    ).reset_index()

    rfm["recency"] = (today - rfm["last_order_date"]).dt.days
    rfm["account_age_days"] = (today - rfm["user_created_at"]).dt.days

    # -------------------- METHOD --------------------
    seg_method = st.selectbox(
        "📊 Select Segmentation Method",
        ["Rule-Based", "KMeans Clustering"]
    )

    # -------------------- RULE BASED --------------------
    if seg_method == "Rule-Based":
        st.subheader("📋 Rule-Based Segmentation")

        rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
        rfm["f_score"] = pd.qcut(
            rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        rfm["m_score"] = pd.qcut(
            rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
        ).astype(int)

        rfm["rfm_score"] = rfm[["r_score", "f_score", "m_score"]].sum(axis=1)

        def label_segment(row):
            if row["rfm_score"] >= 13:
                return "Champions"
            elif row["r_score"] >= 4 and row["f_score"] >= 4:
                return "Loyal Customers"
            elif row["r_score"] >= 4:
                return "Potential Loyalist"
            elif row["r_score"] <= 2 and row["f_score"] <= 2:
                return "At Risk"
            elif row["f_score"] == 1 and row["m_score"] == 1:
                return "Lost"
            else:
                return "Others"

        rfm["segment"] = rfm.apply(label_segment, axis=1)

    # -------------------- KMEANS --------------------
    else:
        st.subheader("🤖 KMeans Clustering")

        features = rfm[["recency", "frequency", "monetary"]]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm["cluster"] = kmeans.fit_predict(scaled)

        centers = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=["recency", "frequency", "monetary"]
        )

        def label_cluster(row):
            if row["recency"] < 30 and row["frequency"] > 8:
                return "Champions"
            elif row["recency"] < 45 and row["frequency"] > 4:
                return "Loyal Customers"
            elif row["recency"] < 60 and row["frequency"] > 2:
                return "Potential Loyalist"
            elif row["recency"] > 90:
                return "Lost"
            else:
                return "At Risk"

        centers["segment"] = centers.apply(label_cluster, axis=1)
        mapping = centers["segment"].to_dict()

        rfm["segment"] = rfm["cluster"].map(mapping)

    # -------------------- BAR CHART --------------------
    st.subheader("📊 Segment Distribution")
    seg_counts = rfm["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "count"]

    chart = alt.Chart(seg_counts).mark_bar().encode(
        x="count",
        y=alt.Y("segment", sort="-x"),
        color=alt.Color("segment", legend=None),
    ).properties(height=300)

    st.altair_chart(chart, use_container_width=True)

    # -------------------- EXPORT BY SEGMENT --------------------
    st.subheader("📥 Export Segmented Users by Segment")

    today_str = datetime.today().strftime("%Y-%m-%d")

    for seg in sorted(rfm["segment"].unique()):
        temp = rfm[rfm["segment"] == seg][["phone", "name"]].rename(
            columns={"phone": "MOBILE", "name": "FIRSTNAME"}
        )

        st.download_button(
            f"Download {seg} Users",
            temp.to_csv(index=False).encode("utf-8"),
            f"{seg.lower().replace(' ', '_')}_{today_str}.csv",
            "text/csv"
        )

    # -------------------- DOWNLOAD ALL --------------------
    st.subheader("📦 Download ALL")

    all_df = rfm[
        ["user_id", "name", "phone", "recency", "frequency", "monetary", "segment"]
    ]

    st.download_button(
        "⬇️ Download ALL Segmented Data",
        all_df.to_csv(index=False).encode("utf-8"),
        f"ALL_SEGMENTS_{today_str}.csv",
        "text/csv"
    )

    # -------------------- PREVIOUS COMPARISON --------------------
    st.subheader("📂 Previous Segment Comparison")

    prev_file = st.file_uploader("Upload previous segmented CSV", type=["csv"])

    if prev_file:
        prev_df = pd.read_csv(prev_file)

        if "user_id" in prev_df.columns and "segment" in prev_df.columns:
            st.success("✅ Previous data loaded")

            # Size Change
            prev_counts = prev_df["segment"].value_counts().reset_index()
            prev_counts.columns = ["segment", "prev_users"]

            curr_counts = rfm["segment"].value_counts().reset_index()
            curr_counts.columns = ["segment", "current_users"]

            merged = pd.merge(prev_counts, curr_counts, on="segment", how="outer").fillna(0)
            merged["change"] = merged["current_users"] - merged["prev_users"]

            st.markdown("### 📊 Segment Size Change")
            st.dataframe(merged)

            # -------- USER MOVEMENT --------
            st.markdown("### 🔁 Where users moved (Previous → Current)")

            movement = prev_df.merge(
                rfm[["user_id", "segment"]],
                on="user_id",
                how="left",
                suffixes=("_previous", "_current")
            )

            movement = movement[movement["segment_previous"] != movement["segment_current"]]

            selected = st.selectbox(
                "Select previous segment",
                sorted(movement["segment_previous"].dropna().unique())
            )

            moved = movement[movement["segment_previous"] == selected]

            moved_summary = (
                moved["segment_current"]
                .value_counts()
                .reset_index()
            )
            moved_summary.columns = ["Moved To", "Users"]

            st.markdown(f"### Users moved from {selected}")
            st.dataframe(moved_summary, use_container_width=True)

            # -------------------- CLEAN SHOW USER LIST --------------------
            if st.checkbox("Show user list"):
                detailed = moved.merge(
                    rfm[["user_id", "name", "phone"]],
                    on="user_id",
                    how="left"
                )

                # ✅ DROP NULL VALUES HERE
                cleaned = detailed.dropna(subset=["user_id", "name", "phone"])

                removed_rows = len(detailed) - len(cleaned)
                st.info(f"Removed {removed_rows} rows with missing user data")

                st.dataframe(
                    cleaned[
                        ["user_id", "name", "phone", "segment_previous", "segment_current"]
                    ],
                    use_container_width=True
                )

                # Optional download
                st.download_button(
                    "Download cleaned moved users",
                    cleaned.to_csv(index=False).encode("utf-8"),
                    f"cleaned_moved_users_{today_str}.csv",
                    "text/csv"
                )

        else:
            st.error("CSV must contain user_id and segment")

    # -------------------- PREVIEW --------------------
    if st.checkbox("👁️ Show Sample Data"):
        st.dataframe(
            rfm[
                [
                    "user_id",
                    "name",
                    "phone",
                    "recency",
                    "frequency",
                    "monetary",
                    "segment",
                ]
            ].head()
        )
