import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

st.set_page_config(page_title="Customer Segmentation", page_icon="🛒", layout="wide")

# Cluster info based on your analysis
CLUSTERS = {
    0: {"name": "🌟 The Best", "desc": "High Income, High Spend, Always accepts campaigns", 
        "action": "Don't change anything; keep them happy.", "color": "#2ca02c"},
    1: {"name": "⚠️ The Struggle", "desc": "Lowest Income, Lowest Spend, Never accepts campaigns",
        "action": "Minimize marketing spend here to save money.", "color": "#d62728"},
    2: {"name": "📈 The Promising", "desc": "Lower Income, but Always accepts campaigns (likely sales/discounts)",
        "action": "Send them coupons and 'Sale' alerts. They respond well to digital marketing.", "color": "#ff7f0e"},
    3: {"name": "💎 The Missed Opportunity", "desc": "High Income, High Spend, Never accepts campaigns",
        "action": "Stop sending emails (they ignore them). Focus on in-store experience or VIP events.", "color": "#1f77b4"}
}

@st.cache_resource
def load_models():
    models = {}

    # External combination models you receive:
    combinations = [
        'RandomForest_Agglomerative',
        'RandomForest_KMeans',
        'RandomForest_GMM',
        'XGBoost_Agglomerative',
        'XGBoost_KMeans',
        'XGBoost_GMM',
        'LightGBM_Agglomerative',
        'LightGBM_KMeans',
        'LightGBM_GMM'
    ]

    for combo in combinations:
        filename = f"{combo}.pkl"
        try:
            models[combo] = pickle.load(open(filename, 'rb'))
        except:
            models[combo] = None

    # Main in‑house RF model (Agglomerative)
    try:
        models['rf_main'] = pickle.load(open('random_forest_main.pkl', 'rb'))
    except:
        models['rf_main'] = None

    # Scaler
    try:
        models['scaler'] = pickle.load(open('scaler.pkl', 'rb'))
    except:
        models['scaler'] = None

    return models



models = load_models()

def compute_features(df):
    """Compute derived features for CSV uploads"""
    df = df.copy()
    # Total spend from individual categories if available
    spend_cols = ['spend_wine', 'spend_fruits', 'spend_meat', 'spend_fish', 'spend_sweets', 'spend_gold']
    if all(c in df.columns for c in spend_cols):
        df['total_spend'] = df[spend_cols].sum(axis=1)
    # Total children
    if 'num_teenagers' in df.columns and 'num_children' in df.columns:
        df['total_children'] = df['num_teenagers'] + df['num_children']
    # Compute ratios
    total = df.get('num_web_purchases', 0) + df.get('num_store_purchases', 0) + df.get('num_catalog_purchases', 0)
    df['OnlineShoppingRatio'] = np.where(total > 0, df.get('num_web_purchases', 0) / total, 0)
    df['spend_per_purchase'] = np.where(total > 0, df.get('total_spend', 0) / total, 0)
    # Recency score
    if 'days_since_last_purchase' in df.columns:
        df['recency_score'] = 1 / (df['days_since_last_purchase'] + 1)
    # Campaign acceptance rate
    campaign_cols = ['accepted_campaign_1', 'accepted_campaign_2', 'accepted_campaign_3', 
                     'accepted_campaign_4', 'accepted_campaign_5', 'accepted_last_campaign']
    if any(c in df.columns for c in campaign_cols):
        df['campaign_acceptance_rate'] = df[[c for c in campaign_cols if c in df.columns]].max(axis=1)
    return df

# The 14 features used for classification
FEATURE_COLS = ['OnlineShoppingRatio', 'spend_per_purchase', 'recency_score',
                'campaign_acceptance_rate', 'total_spend', 'num_web_purchases',
                'num_store_purchases', 'num_catalog_purchases', 'Age', 
                'annual_income', 'education_level', 'total_children',
                'web_visits_last_month', 'customer_since']

def get_features(df):
    """Extract the 14 features needed for classification"""
    for c in FEATURE_COLS:
        if c not in df.columns: 
            df[c] = 0
    return df[FEATURE_COLS]
def cluster(X, method):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # For single prediction, MUST use Agglomerative
    if X_scaled.shape == 1:
        return AgglomerativeClustering(n_clusters=4, linkage='average').fit_predict(X_scaled)
    
    # For multiple samples, can use KMeans or GMM
    if method == "K-Means":
        return KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_scaled)
    elif method == "GMM":
        km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_scaled)
        return GaussianMixture(n_components=4, means_init=km.cluster_centers_, random_state=42).fit_predict(X_scaled)
    else:
        return AgglomerativeClustering(n_clusters=4, linkage='average').fit_predict(X_scaled)
def classify(X, cluster_method, class_method):
    """
    class_method: 'RandomForest', 'XGBoost', 'LightGBM'
    cluster_method: 'Agglomerative', 'KMeans', 'GMM'
    """
    combo_name = f"{class_method}_{cluster_method}"
    model = models.get(combo_name)

    # If external combination model not found:
    if model is None and class_method == 'RandomForest' and cluster_method == 'Agglomerative':
        model = models.get('rf_main')

    return model.predict(X) if model is not None else None

def show_result(cluster_id, title=""):
    c = CLUSTERS[cluster_id]
    st.markdown(f"### {title}{c['name']}")
    st.info(f"**Profile:** {c['desc']}")
    st.success(f"**Recommendation:** {c['action']}")

# === MAIN APP ===
st.title("🛒 Customer Segmentation")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")

    input_mode = st.radio("📥 Data Input", ["Manual", "CSV Upload"])

    st.markdown("---")
    st.subheader("🤖 Model Selection")

    cluster_model = st.selectbox(
        "Clustering Method",
        ["Agglomerative", "KMeans", "GMM"]
    )

    class_model = st.selectbox(
        "Classification Method",
        ["RandomForest", "XGBoost", "LightGBM"]
    )

    st.markdown("---")
    st.info(f"📦 Using: `{class_model}_{cluster_model}.pkl` (or rf_main fallback)")


if input_mode == "Manual":
    st.subheader("Enter Customer Data")
    
    # Row 1: Demographics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age", 18, 100, 35)
    with c2:
        income = st.number_input("Annual Income", 0, 500000, 50000)
    with c3:
        edu = st.selectbox("Education", ["Basic", "2n Cycle", "Graduation", "Master", "PhD"])
        edu_map = {"Basic": 1, "2n Cycle": 2, "Graduation": 3, "Master": 4, "PhD": 5}
        edu = edu_map[edu]
    with c4:
        children = st.number_input("Total Children", 0, 10, 1)
    
    # Row 2: Spending & Purchases
    st.markdown("**Spending & Purchases**")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        total_spend = st.number_input("Total Spend ($)", 0, 100000, 500)
    with s2:
        web = st.number_input("Web Purchases", 0, 100, 5)
    with s3:
        catalog = st.number_input("Catalog Purchases", 0, 100, 2)
    with s4:
        store = st.number_input("Store Purchases", 0, 100, 8)
    
    # Row 3: Activity
    st.markdown("**Activity**")
    a1, a2, a3 = st.columns(3)
    with a1:
        visits = st.number_input("Web Visits (month)", 0, 100, 5)
    with a2:
        days = st.number_input("Days Since Purchase", 0, 365, 30)
    with a3:
        since = st.number_input("Customer Since (days)", 0, 5000, 365)
    
    # Row 4: Campaign
    st.markdown("**Campaign Response**")
    campaign = st.selectbox("Accepted Any Campaign?", ["No", "Yes"])
    campaign_rate = 1.0 if campaign == "Yes" else 0.0

    if st.button("🔍 Analyze", type="primary"):
        total_purch = web + catalog + store
        
        # Build the 14-feature dataframe
        df = pd.DataFrame([{
            'OnlineShoppingRatio': web / total_purch if total_purch > 0 else 0,
            'spend_per_purchase': total_spend / total_purch if total_purch > 0 else 0,
            'recency_score': 1 / (days + 1),
            'campaign_acceptance_rate': campaign_rate,
            'total_spend': total_spend,
            'num_web_purchases': web,
            'num_store_purchases': store,
            'num_catalog_purchases': catalog,
            'Age': age,
            'annual_income': income,
            'education_level': edu,
            'total_children': children,
            'web_visits_last_month': visits,
            'customer_since': since
        }])
    
        st.markdown("---")
        st.subheader("📊 Result")
    
        # Classify using the 14 features
        pred = classify(get_features(df), cluster_model, class_model)
    
        if pred is not None:
            show_result(int(pred[0]), f"{class_model} ({cluster_model}): ")
        else:
            st.error(f"⚠️ Model not found: `{class_model}_{cluster_model}.pkl`")
            st.info("Make sure the file exists in your project folder!")

else:  # CSV Upload
    file = st.file_uploader("Upload CSV", type=['csv'])
    if file:
        df = compute_features(pd.read_csv(file))
        st.success(f"✓ Loaded {len(df)} customers")
        
        if st.button("🔍 Analyze All", type="primary"):
            # Use classification from selected combination
            pred = classify(get_features(df), cluster_model, class_model)
            if pred is not None:
                df['Cluster'] = pred
            else:
                # Fallback to clustering if classifier not found
                df['Cluster'] = cluster(get_features(df), cluster_model)
            
            st.markdown("---")
            tab1, tab2 = st.tabs(["📊 Summary", "📋 Data"])
            
            with tab1:
                st.metric("Total Customers", len(df))
                
                cols = st.columns(4)
                for i in range(4):
                    count = (df['Cluster'] == i).sum()
                    if count > 0:
                        cols[i].metric(
                            CLUSTERS[i]['name'],
                            f"{count} customers",
                            f"{count/len(df)*100:.1f}%"
                        )
                
                st.markdown("---")
                for i in range(4):
                    count = (df['Cluster'] == i).sum()
                    if count > 0:
                        with st.expander(f"{CLUSTERS[i]['name']} - {count} customers"):
                            st.write(f"**Profile:** {CLUSTERS[i]['desc']}")
                            st.write(f"**Action:** {CLUSTERS[i]['action']}")
            
            with tab2:
                df['Cluster_Name'] = df['Cluster'].map(lambda x: CLUSTERS[x]['name'])
                st.dataframe(df)
                st.download_button("📥 Download", df.to_csv(index=False), "results.csv", "text/csv")
