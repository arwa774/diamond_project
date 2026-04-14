"""
app.py — Dashboard Streamlit : Prédiction du Prix des Diamants
Auteur : Hadj Mbarek Arwa
"""


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


# ── Import du module de preprocessing ────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import charger_et_nettoyer, encoder_variables_ordinales, creer_features


# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💎 Diamond Price Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Thème Plotly personnalisé ─────────────────────────────────────────────────
PLOTLY_THEME = "plotly_white"
COLOR_PRIMARY   = "#534AB7"
COLOR_SECONDARY = "#7F77DD"
COLOR_ACCENT    = "#D85A30"
COLOR_PALETTE   = ["#534AB7", "#D85A30", "#1D9E75", "#BA7517", "#993556", "#185FA5"]


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #f8f7ff; }
    .main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; text-align: center; }
    .section-title { font-size: 1.2rem; font-weight: 600; color: #534AB7;
                     border-left: 4px solid #534AB7; padding-left: 10px; margin: 1rem 0 0.5rem; }
    .predict-box { background: #EEEDFE; border: 1px solid #AFA9EC; border-radius: 16px;
                   padding: 1.5rem; text-align: center; margin: 1rem 0; }
    .predict-price { font-size: 3rem; font-weight: 800; color: #26215C; }
    .predict-range { font-size: 0.85rem; color: #7F77DD; margin-top: 4px; }
    .stButton > button { width: 100%; background: #534AB7; color: white; border: none;
                         border-radius: 10px; padding: 0.75rem; font-size: 1rem;
                         font-weight: 600; margin-top: 0.5rem; }
    .stButton > button:hover { background: #3C3489; }
    div[data-testid="metric-container"] { background: #f8f7ff; border-radius: 10px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)




# ════════════════════════════════════════════════════════════════════════════════
#   CHARGEMENT DONNÉES & MODÈLE
# ════════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    # Nettoyage via preprocessing.py — évite la duplication de code
    df = charger_et_nettoyer(r"C:\Users\TUF\Downloads\diamond_project\diamond_project\data\diamonds.csv")
    # On ajoute le volume ici pour les visualisations (avant encodage)
    df["volume"] = df["x"] * df["y"] * df["z"]
    return df


@st.cache_resource
def load_model():
    path = "model/best_model_optimise.pkl"
    if not os.path.exists(path):
        st.error("Modèle introuvable ! Lancez d'abord : python train_model.py")
        st.stop()
    return joblib.load(path)


@st.cache_data
def get_predictions(_model, df):
    # Encodage + feature engineering via preprocessing.py
    df2 = encoder_variables_ordinales(df.drop(columns=["volume"]))
    df2 = creer_features(df2)
    feat_cols = ["carat","depth","table","x","y","z", "volume" ,"carat_per_volume",
                 "cut_encoded","color_encoded","clarity_encoded"
                 ]
    X = df2[feat_cols]
    y = df2["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = _model.predict(X_test)
    return y_test.values, y_pred, X_test, _model.feature_importances_, feat_cols


df    = load_data()
model = load_model()
y_test, y_pred, X_test, importances, feat_cols = get_predictions(model, df)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mae  = np.mean(np.abs(y_test - y_pred))




# ════════════════════════════════════════════════════════════════════════════════
#   SIDEBAR — Formulaire de prédiction
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💎 Prédire un prix")
    st.markdown("---")


    carat = st.slider("Carat (poids)", 0.20, 5.01, 1.00, 0.01)
    cut   = st.selectbox("Cut", ["Fair","Good","Very Good","Premium","Ideal"], index=4)
    color = st.selectbox("Color", ["J","I","H","G","F","E","D"], index=6)
    clarity = st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"], index=7)


    st.markdown("**Dimensions**")
    col_a, col_b = st.columns(2)
    with col_a:
        x = st.number_input("X (mm)", 0.0, 11.0, 4.5, 0.1)
        y_val = st.number_input("Y (mm)", 0.0, 11.0, 4.5, 0.1)
    with col_b:
        z = st.number_input("Z (mm)", 0.0, 7.0, 2.8, 0.1)
        depth = st.number_input("Depth %", 43.0, 79.0, 61.5, 0.1)


    table = st.slider("Table %", 43.0, 95.0, 57.0, 1.0)


    predict_clicked = st.button("Estimer le prix")


    st.markdown("---")
    st.markdown("**Modèle : Random Forest**")
    st.caption(f"R² = {r2:.4f} · RMSE = ${rmse:,.0f}")
    st.caption("Dataset : 53 940 diamants (Kaggle)")
    st.caption("Auteure : Hadj Mbarek Arwa · 2026")




# ── Prédiction ─────────────────────────────────────────────────────────────────
def predict(carat, cut, color, clarity, depth, table, x, y, z):
    row = pd.DataFrame([{
        "carat":   float(carat),
        "depth":   float(depth),
        "table":   float(table),
        "x":       float(x),
        "y":       float(y),
        "z":       float(z),
        "cut":     cut,
        "color":   color,
        "clarity": clarity,
        "price":   0,
    }])
    row = encoder_variables_ordinales(row)
    row = creer_features(row)
    feat_cols = ["carat","depth","table","x","y","z",
                 "volume","carat_per_volume",
                 "cut_encoded","color_encoded","clarity_encoded"]
    return max(0, round(float(model.predict(row[feat_cols])[0]), 2))




# ════════════════════════════════════════════════════════════════════════════════
#   MAIN LAYOUT
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<h1 class="main-title">💎 Diamond Price — Data Science Dashboard</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#6c757d;margin-bottom:1.5rem'>Exploration · Modélisation · Prédiction</p>", unsafe_allow_html=True)


# ── KPI ROW ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Diamants analysés", f"{len(df):,}")
k2.metric("Prix moyen", f"${df['price'].mean():,.0f}")
k3.metric("Prix médian", f"${df['price'].median():,.0f}")
k4.metric("R² du modèle", f"{r2:.4f}")
k5.metric("RMSE", f"${rmse:,.0f}")




# ── RÉSULTAT PRÉDICTION ───────────────────────────────────────────────────────
if predict_clicked:
    prix = predict(carat, cut, color, clarity, depth, table, x, y_val, z)
    marge = prix * 0.15
    st.markdown(f"""
    <div class="predict-box">
        <div style="font-size:0.9rem;color:#534AB7;margin-bottom:4px">Prix estimé pour votre diamant</div>
        <div class="predict-price">${prix:,.0f}</div>
        <div class="predict-range">Fourchette : ${prix-marge:,.0f} — ${prix+marge:,.0f}</div>
    </div>""", unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Carat", carat)
    pc2.metric("Cut", cut)
    pc3.metric("Color", color)
    pc4.metric("Clarity", clarity)
    if prix < 1000:   st.info("Gamme entrée de gamme (< $1,000)")
    elif prix < 5000: st.info("Gamme milieu de gamme ($1,000 — $5,000)")
    elif prix < 10000:st.success("Gamme haut de gamme ($5,000 — $10,000)")
    else:             st.success("Gamme luxe (> $10,000)")
    st.divider()




# ════════════════════════════════════════════════════════════════════════════════
#   SECTION 1 — DISTRIBUTION DES DONNÉES
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Distribution des données</div>', unsafe_allow_html=True)


row1_col1, row1_col2 = st.columns(2)


with row1_col1:
    # 1. Distribution du prix (histogramme + courbe KDE)
    fig_price = go.Figure()
    fig_price.add_trace(go.Histogram(
        x=df["price"], nbinsx=80,
        name="Prix",
        marker_color=COLOR_PRIMARY,
        opacity=0.8,
        histnorm="probability density",
    ))
    fig_price.update_layout(
        title="Distribution du prix des diamants",
        xaxis_title="Prix ($)",
        yaxis_title="Densité",
        template=PLOTLY_THEME,
        showlegend=False,
        height=320,
        margin=dict(t=40, b=40, l=40, r=20),
    )
    fig_price.add_vline(x=df["price"].mean(), line_dash="dash",
                        line_color=COLOR_ACCENT, annotation_text=f"Moyenne ${df['price'].mean():,.0f}")
    st.plotly_chart(fig_price, use_container_width=True)


with row1_col2:
    # 2. Distribution du carat (boxplot par cut)
    cut_order = ["Fair","Good","Very Good","Premium","Ideal"]
    fig_box = px.box(
        df, x="cut", y="price",
        category_orders={"cut": cut_order},
        color="cut",
        color_discrete_sequence=COLOR_PALETTE,
        title="Distribution du prix par qualité de taille (Cut)",
        labels={"price": "Prix ($)", "cut": "Cut"},
        template=PLOTLY_THEME,
    )
    fig_box.update_layout(showlegend=False, height=320,
                          margin=dict(t=40, b=40, l=40, r=20))
    st.plotly_chart(fig_box, use_container_width=True)




# ════════════════════════════════════════════════════════════════════════════════
#   SECTION 2 — RELATIONS ENTRE VARIABLES
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Relations entre variables</div>', unsafe_allow_html=True)


row2_col1, row2_col2 = st.columns(2)


with row2_col1:
    # 3. Scatter plot : Carat vs Prix (coloré par Cut)
    sample = df.sample(5000, random_state=42)
    fig_scatter = px.scatter(
        sample, x="carat", y="price",
        color="cut",
        category_orders={"cut": cut_order},
        color_discrete_sequence=COLOR_PALETTE,
        title="Carat vs Prix — coloré par Cut",
        labels={"carat": "Carat (poids)", "price": "Prix ($)", "cut": "Cut"},
        template=PLOTLY_THEME,
        opacity=0.6,
        size_max=6,
    )
    fig_scatter.update_layout(height=340, margin=dict(t=40, b=40, l=40, r=20))
    st.plotly_chart(fig_scatter, use_container_width=True)


with row2_col2:
    # 4. Heatmap de corrélation
    num_cols = ["carat","depth","table","price","x","y","z","volume"]
    corr = df[num_cols].corr().round(2)
    fig_heat = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        textfont={"size": 11},
        colorbar=dict(title="Corrélation"),
    ))
    fig_heat.update_layout(
        title="Matrice de corrélation",
        template=PLOTLY_THEME,
        height=340,
        margin=dict(t=40, b=40, l=40, r=40),
    )
    st.plotly_chart(fig_heat, use_container_width=True)




# ════════════════════════════════════════════════════════════════════════════════
#   SECTION 3 — ANALYSE PAR CATÉGORIE
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Analyse par catégorie</div>', unsafe_allow_html=True)


row3_col1, row3_col2, row3_col3 = st.columns(3)


with row3_col1:
    # 5. Prix moyen par Cut
    cut_price = df.groupby("cut")["price"].mean().reindex(cut_order).reset_index()
    fig_cut = px.bar(
        cut_price, x="cut", y="price",
        color="price",
        color_continuous_scale=["#EEEDFE","#534AB7"],
        title="Prix moyen par Cut",
        labels={"price": "Prix moyen ($)", "cut": "Cut"},
        template=PLOTLY_THEME,
        text_auto="$.0f",
    )
    fig_cut.update_layout(showlegend=False, coloraxis_showscale=False,
                          height=300, margin=dict(t=40,b=40,l=40,r=20))
    fig_cut.update_traces(textposition="outside")
    st.plotly_chart(fig_cut, use_container_width=True)


with row3_col2:
    # 6. Prix moyen par Color
    color_order = ["J","I","H","G","F","E","D"]
    color_price = df.groupby("color")["price"].mean().reindex(color_order).reset_index()
    fig_color = px.bar(
        color_price, x="color", y="price",
        color="price",
        color_continuous_scale=["#FAECE7","#D85A30"],
        title="Prix moyen par Color",
        labels={"price": "Prix moyen ($)", "color": "Color"},
        template=PLOTLY_THEME,
        text_auto="$.0f",
    )
    fig_color.update_layout(showlegend=False, coloraxis_showscale=False,
                            height=300, margin=dict(t=40,b=40,l=40,r=20))
    fig_color.update_traces(textposition="outside")
    st.plotly_chart(fig_color, use_container_width=True)


with row3_col3:
    # 7. Prix moyen par Clarity
    clarity_order = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
    clarity_price = df.groupby("clarity")["price"].mean().reindex(clarity_order).reset_index()
    fig_clarity = px.bar(
        clarity_price, x="clarity", y="price",
        color="price",
        color_continuous_scale=["#E1F5EE","#1D9E75"],
        title="Prix moyen par Clarity",
        labels={"price": "Prix moyen ($)", "clarity": "Clarity"},
        template=PLOTLY_THEME,
        text_auto="$.0f",
    )
    fig_clarity.update_layout(showlegend=False, coloraxis_showscale=False,
                              height=300, margin=dict(t=40,b=40,l=40,r=20))
    fig_clarity.update_traces(textposition="outside")
    st.plotly_chart(fig_clarity, use_container_width=True)




# ════════════════════════════════════════════════════════════════════════════════
#   SECTION 4 — PERFORMANCE DU MODÈLE
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Performance du modèle Random Forest</div>', unsafe_allow_html=True)


row4_col1, row4_col2 = st.columns(2)


with row4_col1:
    # 8. Actual vs Predicted scatter
    idx = np.random.choice(len(y_test), size=3000, replace=False)
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(
        x=y_test[idx], y=y_pred[idx],
        mode="markers",
        marker=dict(color=COLOR_PRIMARY, opacity=0.4, size=4),
        name="Prédictions",
    ))
    max_val = max(y_test.max(), y_pred.max())
    fig_avp.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color=COLOR_ACCENT, dash="dash", width=2),
        name="Prédiction parfaite",
    ))
    fig_avp.update_layout(
        title=f"Réel vs Prédit  (R² = {r2:.4f})",
        xaxis_title="Prix réel ($)",
        yaxis_title="Prix prédit ($)",
        template=PLOTLY_THEME,
        height=340,
        margin=dict(t=40,b=40,l=40,r=20),
    )
    st.plotly_chart(fig_avp, use_container_width=True)


with row4_col2:
    # 9. Distribution des résidus
    residuals = y_test - y_pred
    fig_res = go.Figure()
    fig_res.add_trace(go.Histogram(
        x=residuals, nbinsx=80,
        marker_color=COLOR_SECONDARY,
        opacity=0.85,
        name="Résidus",
    ))
    fig_res.add_vline(x=0, line_dash="dash", line_color=COLOR_ACCENT,
                      annotation_text="Erreur = 0")
    fig_res.update_layout(
        title=f"Distribution des résidus  (MAE = ${mae:,.0f})",
        xaxis_title="Erreur = Réel − Prédit ($)",
        yaxis_title="Fréquence",
        template=PLOTLY_THEME,
        showlegend=False,
        height=340,
        margin=dict(t=40,b=40,l=40,r=20),
    )
    st.plotly_chart(fig_res, use_container_width=True)




# ════════════════════════════════════════════════════════════════════════════════
#   SECTION 5 — FEATURE IMPORTANCE + VOLUME
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Importance des variables & Volume</div>', unsafe_allow_html=True)


row5_col1, row5_col2 = st.columns(2)


with row5_col1:
    # 10. Feature Importance (horizontal bar chart)
    fi_df = pd.DataFrame({
        "feature": feat_cols,
        "importance": importances,
    }).sort_values("importance", ascending=True)


    fig_fi = px.bar(
        fi_df, x="importance", y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#EEEDFE","#534AB7"],
        title="Importance des features (Random Forest)",
        labels={"importance": "Importance", "feature": "Variable"},
        template=PLOTLY_THEME,
        text_auto=".3f",
    )
    fig_fi.update_layout(showlegend=False, coloraxis_showscale=False,
                         height=360, margin=dict(t=40,b=40,l=100,r=20))
    fig_fi.update_traces(textposition="outside")
    st.plotly_chart(fig_fi, use_container_width=True)


with row5_col2:
    # 11. Volume vs Prix (bubble chart coloré par carat)
    sample2 = df.sample(3000, random_state=99)
    fig_bubble = px.scatter(
        sample2, x="volume", y="price",
        color="carat",
        color_continuous_scale=["#EEEDFE","#534AB7","#26215C"],
        title="Volume vs Prix — taille = carat",
        labels={"volume": "Volume (mm³)", "price": "Prix ($)", "carat": "Carat"},
        template=PLOTLY_THEME,
        size="carat",
        size_max=12,
        opacity=0.6,
    )
    fig_bubble.update_layout(height=360, margin=dict(t=40,b=40,l=40,r=20))
    st.plotly_chart(fig_bubble, use_container_width=True)




# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style="text-align:center;color:#6c757d;font-size:0.8rem">
    💎 Diamond Price Dashboard · Hadj Mbarek Arwa · 2026 ·
    Modèle Random Forest · Dataset Kaggle (53,940 diamants)
</p>
""", unsafe_allow_html=True)



