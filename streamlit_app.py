import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import altair as alt

st.set_page_config(page_title="🧠 RFM + KMeans Segmentation", layout="wide")
st.title("🧠 RFM Segmentation Dashboard (with KMeans Option)")

st.markdown("Upload a single `orders.csv` file with the following columns:")
st.code("user_id, created_at, total, sub_total, discount, coupon_id, payment_method, actual_qty, user_created_at, name, phone")

uploaded_file = st.file_uploader("📤 Upload orders.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['user_created_at'] = pd.to_datetime(df['user_created_at'])
    today = pd.to_datetime("today")

    # RFM Table with phone and name
    rfm = df.groupby('user_id').agg(
        last_order_date=('created_at', 'max'),
        frequency=('user_id', 'count'),
        monetary=('total', 'sum'),
        user_created_at=('user_created_at', 'min'),
        name=('name', 'first'),
        phone=('phone', 'first')
    ).reset_index()
    rfm['recency'] = (today - rfm['last_order_date']).dt.days
    rfm['account_age_days'] = (today - rfm['user_created_at']).dt.days

    seg_method = st.selectbox("📊 Select Segmentation Method", ["Rule-Based", "KMeans Clustering"])

    if seg_method == "Rule-Based":
        st.subheader("📋 Rule-Based Segmentation")

        rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm['rfm_score'] = rfm[['r_score', 'f_score', 'm_score']].sum(axis=1)

        def label_segment(row):
            if row['rfm_score'] >= 13:
                return 'Champions'
            elif row['r_score'] >= 4 and row['f_score'] >= 4:
                return 'Loyal Customers'
            elif row['r_score'] >= 4:
                return 'Potential Loyalist'
            elif row['r_score'] <= 2 and row['f_score'] <= 2:
                return 'At Risk'
            elif row['f_score'] == 1 and row['m_score'] == 1:
                return 'Lost'
            else:
                return 'Others'

        rfm['segment'] = rfm.apply(label_segment, axis=1)

    elif seg_method == "KMeans Clustering":
        st.subheader("🤖 KMeans Clustering Segmentation")

        features = rfm[['recency', 'frequency', 'monetary']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm['cluster'] = kmeans.fit_predict(features_scaled)

        # Recover unscaled centers
        unscaled_centers = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=['recency', 'frequency', 'monetary']
        )

        def label_cluster(row):
            if row['recency'] < 30 and row['frequency'] > 8 and row['monetary'] > 3000:
                return 'Champions'
            elif row['recency'] < 45 and row['frequency'] > 4:
                return 'Loyal Customers'
            elif row['recency'] < 60 and row['frequency'] > 2:
                return 'Potential Loyalist'
            elif row['recency'] > 90 and row['frequency'] <= 1:
                return 'Lost'
            elif row['recency'] > 60 and row['frequency'] <= 2:
                return 'At Risk'
            else:
                return 'Others'

        unscaled_centers['segment'] = unscaled_centers.apply(label_cluster, axis=1)
        cluster_id_to_name = unscaled_centers['segment'].to_dict()
        rfm['segment'] = rfm['cluster'].map(cluster_id_to_name)

    # --- Segment Distribution ---
    st.subheader("📊 Segment Distribution")
    seg_counts = rfm['segment'].value_counts().reset_index()
    seg_counts.columns = ['segment', 'count']
    bar_chart = alt.Chart(seg_counts).mark_bar().encode(
        x='count:Q',
        y=alt.Y('segment:N', sort='-x'),
        color=alt.Color('segment:N', legend=None)
    ).properties(
        width=600,
        height=300,
        title="Customer Segment Distribution"
    )
    st.altair_chart(bar_chart, use_container_width=True)

    # --- Export Segments Separately ---
    st.subheader("📥 Export Segmented Users by Segment")
    
    segment_list = rfm['segment'].unique()
    segment_list.sort()
    today_str = datetime.today().strftime('%Y-%m-%d')
    
    for segment in segment_list:
        seg_df = rfm[rfm['segment'] == segment][['phone', 'name']].rename(columns={
            'phone': 'MOBILE',
            'name': 'FIRSTNAME'
        })
    
        seg_csv = seg_df.to_csv(index=False).encode('utf-8')
        file_name = f"{segment.replace(' ', '_').lower()}_users_{today_str}.csv"
    
        st.download_button(
            label=f"Download {segment} Users",
            data=seg_csv,
            file_name=file_name,
            mime="text/csv",
            key=f"download_{segment}"
        )


    # --- Previous Segment Comparison ---
    st.subheader("📂 Previous Segment Comparison (Optional)")
    prev_file = st.file_uploader("Upload previous segmented CSV (user_id, segment)", type=["csv"], key="prev_file_upload")

    if prev_file:
        prev_df = pd.read_csv(prev_file)
        if 'user_id' in prev_df.columns and 'segment' in prev_df.columns:
            st.success("✅ Previous segmented data uploaded!")

            # --- Segment Size Change ---
            prev_counts = prev_df['segment'].value_counts().reset_index()
            prev_counts.columns = ['segment', 'prev_users']

            current_counts = rfm['segment'].value_counts().reset_index()
            current_counts.columns = ['segment', 'current_users']

            merged_counts = pd.merge(prev_counts, current_counts, on='segment', how='outer').fillna(0)
            merged_counts['change_in_users'] = merged_counts['current_users'] - merged_counts['prev_users']
            merged_counts['Change Rate (%)'] = (merged_counts['change_in_users'] / merged_counts['prev_users'].replace(0, 1) * 100).round(2)

            st.markdown("### 📊 Segment Size Change from Previous Run")
            st.dataframe(merged_counts)

            # --- Reorder Analysis After Date Range ---
            st.markdown("### 🔄 Reorder Conversion by Segment")

            st.markdown("#### 📅 Select Date Range to Track Reorders")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start date", key="start_date_picker")
            with col2:
                end_date = st.date_input("End date", value=datetime.today(), key="end_date_picker")

            # Filter users from previous segments who ordered within the range
            prev_segment_users = prev_df[['user_id', 'segment']]
            df['created_at'] = pd.to_datetime(df['created_at'])
            mask = (df['created_at'].dt.date >= start_date) & (df['created_at'].dt.date <= end_date)
            reorder_users = df[mask][['user_id']].drop_duplicates()

            reordered_df = prev_segment_users.merge(reorder_users, on='user_id', how='inner')

            reorder_summary = reordered_df.groupby('segment').agg(
                reordered_users=('user_id', 'nunique')
            ).reset_index()

            reorder_summary = pd.merge(prev_counts, reorder_summary, on='segment', how='left').fillna(0)
            reorder_summary['Conversion Rate (%)'] = (reorder_summary['reordered_users'] / reorder_summary['prev_users'] * 100).round(2)

            st.dataframe(reorder_summary)
        else:
            st.warning("⚠️ Previous file must contain `user_id` and `segment` columns.")

    # --- Optional Preview ---
    if st.checkbox("👁️ Show Sample Data"):
        st.dataframe(rfm[['user_id', 'name', 'phone', 'recency', 'frequency', 'monetary', 'segment']].head())
