"""
Phishing Detection Dashboard
------------------------------
Matches exactly: phishing_xgboost_and_randomforest_.ipynb
 
Files needed in same folder:
  - phishing_model_retrained.pkl
  - rf_model_retrained.pkl
  - dataset_full.csv
  - dataset_small.csv
 
Run:
  streamlit run dashboard.py
"""
 
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve,
    accuracy_score
)
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings("ignore")
 
 
# -------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------
 
st.set_page_config(
    page_title = "Phishing Detection",
    page_icon  = "🛡️",
    layout     = "wide"
)
 
 
# -------------------------------------------------------
# LOAD MODELS AND DATA
# -------------------------------------------------------
 
@st.cache_resource
def load_model():
    try:
        return joblib.load("phishing_model_retrained.pkl")
    except FileNotFoundError:
        st.error("phishing_model_retrained.pkl not found. Run your notebook first.")
        st.stop()
 
@st.cache_resource
def load_rf_model():
    try:
        return joblib.load("rf_model_retrained.pkl")
    except FileNotFoundError:
        st.error("rf_model_retrained.pkl not found. Run your notebook first.")
        st.stop()
 
@st.cache_data
def load_data():
    try:
        full  = pd.read_csv("dataset_full.csv")
        small = pd.read_csv("dataset_small.csv")
        return full, small
    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        st.stop()
 
model                       = load_model()
rf_model                    = load_rf_model()
dataset_full, dataset_small = load_data()
 
TARGET      = "phishing"
X_full      = dataset_full.drop(columns=[TARGET])
y_full      = dataset_full[TARGET]
X_small     = dataset_small.drop(columns=[TARGET])
y_small     = dataset_small[TARGET]
shared_cols = [c for c in X_full.columns if c in X_small.columns]
X_full      = X_full[shared_cols]
X_small     = X_small[shared_cols]
 
# same split as notebook
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y_full
)
 
# predictions — same as notebook
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
rf_pred      = rf_model.predict(X_test)
rf_pred_prob = rf_model.predict_proba(X_test)[:, 1]
 
# after retraining predictions
y_pred_new  = model.predict(X_small)
rf_pred_new = rf_model.predict(X_small)
 
 
# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
 
st.sidebar.title("🛡️ Phishing Detector")
st.sidebar.markdown("XGBoost + Random Forest + KS Drift Analysis")
st.sidebar.markdown("---")
 
page = st.sidebar.radio(
    "Go to",
    ["🏠  Overview", "📊  Model Performance", "🌊  Drift Analysis", "🔗  URL Checker"]
)
 
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Training samples** : `{len(dataset_full):,}`")
st.sidebar.markdown(f"**Test samples**     : `{len(X_test):,}`")
st.sidebar.markdown(f"**Features**         : `{len(shared_cols)}`")
st.sidebar.markdown(f"**XGBoost accuracy** : `{accuracy_score(y_test, y_pred):.2%}`")
st.sidebar.markdown(f"**RF accuracy**      : `{accuracy_score(y_test, rf_pred):.2%}`")
st.sidebar.markdown(f"**Winner**           : `XGBoost ✅`")
 
 
# -------------------------------------------------------
# PAGE 1 — OVERVIEW
# -------------------------------------------------------
 
if "Overview" in page:
 
    st.title("🛡️ Phishing Detection Dashboard")
    st.markdown("Built with **XGBoost** and **Random Forest** + **Kolmogorov-Smirnov drift analysis**.")
    st.markdown("---")
 
    drifted_count = sum(
        1 for col in shared_cols
        if ks_2samp(X_full[col], X_small[col])[1] < 0.05
    )
 
    # 5 metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("XGBoost Accuracy",  f"{accuracy_score(y_test, y_pred):.2%}")
    c2.metric("RF Accuracy",       f"{accuracy_score(y_test, rf_pred):.2%}")
    c3.metric("Training Samples",  f"{len(dataset_full):,}")
    c4.metric("Features Drifted",  f"{drifted_count} / {len(shared_cols)}")
    c5.metric("Drift Score",       f"{drifted_count / len(shared_cols):.0%}")
 
    st.markdown("---")
 
    # class distribution charts
    col_a, col_b = st.columns(2)
 
    with col_a:
        st.subheader("Class distribution — dataset_full")
        counts = dataset_full[TARGET].value_counts().rename(index={0: "Legit", 1: "Phishing"})
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(counts.index, counts.values, color=["steelblue", "crimson"], width=0.5)
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=11)
        st.pyplot(fig)
        plt.close()
 
    with col_b:
        st.subheader("Class distribution — dataset_small")
        counts2 = dataset_small[TARGET].value_counts().rename(index={0: "Legit", 1: "Phishing"})
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.bar(counts2.index, counts2.values, color=["steelblue", "crimson"], width=0.5)
        ax2.set_ylabel("Count")
        for i, v in enumerate(counts2.values):
            ax2.text(i, v + 200, f"{v:,}", ha="center", fontsize=11)
        st.pyplot(fig2)
        plt.close()
 
    st.markdown("---")
 
    # model comparison chart — same as notebook STEP 4D
    st.subheader("Model Comparison — XGBoost vs Random Forest")
 
    xgb_acc  = accuracy_score(y_test, y_pred)
    rf_acc   = accuracy_score(y_test, rf_pred)
    xgb_f1   = classification_report(y_test, y_pred,  output_dict=True, labels=[0,1])["weighted avg"]["f1-score"]
    rf_f1    = classification_report(y_test, rf_pred, output_dict=True, labels=[0,1])["weighted avg"]["f1-score"]
    xgb_prec = classification_report(y_test, y_pred,  output_dict=True, labels=[0,1])["weighted avg"]["precision"]
    rf_prec  = classification_report(y_test, rf_pred, output_dict=True, labels=[0,1])["weighted avg"]["precision"]
    xgb_rec  = classification_report(y_test, y_pred,  output_dict=True, labels=[0,1])["weighted avg"]["recall"]
    rf_rec   = classification_report(y_test, rf_pred, output_dict=True, labels=[0,1])["weighted avg"]["recall"]
 
    metrics  = ["Accuracy", "F1 Score", "Precision", "Recall"]
    xgb_vals = [xgb_acc, xgb_f1, xgb_prec, xgb_rec]
    rf_vals  = [rf_acc,  rf_f1,  rf_prec,  rf_rec]
    x = np.arange(len(metrics))
 
    fig_cmp, ax_cmp = plt.subplots(figsize=(9, 5))
    bars1 = ax_cmp.bar(x - 0.2, xgb_vals, 0.35, label="XGBoost",       color="steelblue")
    bars2 = ax_cmp.bar(x + 0.2, rf_vals,  0.35, label="Random Forest", color="darkorange")
    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels(metrics)
    ax_cmp.set_ylim(0.85, 1.02)
    ax_cmp.set_ylabel("Score")
    ax_cmp.set_title("Model Comparison — XGBoost vs Random Forest")
    ax_cmp.legend()
    for bar in bars1:
        ax_cmp.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{bar.get_height():.2%}",
                    ha="center", fontsize=9, color="steelblue")
    for bar in bars2:
        ax_cmp.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{bar.get_height():.2%}",
                    ha="center", fontsize=9, color="darkorange")
    st.pyplot(fig_cmp)
    plt.close()
 
    winner = "XGBoost ✅" if xgb_acc >= rf_acc else "Random Forest ✅"
    st.info(f"**Winner : {winner}** — XGBoost: {xgb_acc:.2%}  |  Random Forest: {rf_acc:.2%}")
 
    st.markdown("---")
 
    # static vs retrained — same as notebook STEP 13
    st.subheader("Static vs Retrained Model Comparison")
 
    xgb_before = accuracy_score(y_test, y_pred)
    rf_before  = accuracy_score(y_test, rf_pred)
    xgb_after  = accuracy_score(y_small, y_pred_new)
    rf_after   = accuracy_score(y_small, rf_pred_new)
 
    models_list = ["XGBoost", "Random Forest"]
    before = [xgb_before, rf_before]
    after  = [xgb_after,  rf_after]
    x2 = np.arange(len(models_list))
 
    fig_sv, ax_sv = plt.subplots(figsize=(8, 5))
    b1 = ax_sv.bar(x2 - 0.2, before, 0.35, label="Before Retraining", color="steelblue")
    b2 = ax_sv.bar(x2 + 0.2, after,  0.35, label="After Retraining",  color="crimson")
    ax_sv.set_xticks(x2)
    ax_sv.set_xticklabels(models_list)
    ax_sv.set_ylim(0.85, 1.02)
    ax_sv.set_ylabel("Accuracy")
    ax_sv.set_title("Static vs Retrained Model Comparison")
    ax_sv.legend()
    for bar in b1:
        ax_sv.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.002,
                   f"{bar.get_height():.2%}",
                   ha="center", fontsize=9, color="steelblue")
    for bar in b2:
        ax_sv.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.002,
                   f"{bar.get_height():.2%}",
                   ha="center", fontsize=9, color="crimson")
    st.pyplot(fig_sv)
    plt.close()
 
    st.markdown("---")
    st.subheader("What this system does")
    st.markdown("""
    - Analyses **111 URL-based features** (length, dots, slashes, redirects, SSL status etc.)
    - Trains **XGBoost** and **Random Forest** and compares both
    - Classifies each URL as **Legit (0)** or **Phishing (1)**
    - Detects **data drift** using the KS test when new URL patterns emerge
    - **Automatically retrains** both models when drift is found
    - Models saved via **joblib** — no retraining needed on next load
    """)
 
 
# -------------------------------------------------------
# PAGE 2 — MODEL PERFORMANCE
# -------------------------------------------------------
 
elif "Performance" in page:
 
    st.title("📊 Model Performance")
    st.markdown("Evaluation on the held-out **test set** — 20% of dataset_full = 17,730 samples.")
    st.markdown("---")
 
    xgb_report = classification_report(y_test, y_pred,  target_names=["Legit", "Phishing"], labels=[0, 1], output_dict=True)
    rf_report  = classification_report(y_test, rf_pred, target_names=["Legit", "Phishing"], labels=[0, 1], output_dict=True)
 
    # XGBoost metrics
    st.subheader("XGBoost")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.2%}")
    c2.metric("Precision", f"{xgb_report['Phishing']['precision']:.2%}")
    c3.metric("Recall",    f"{xgb_report['Phishing']['recall']:.2%}")
    c4.metric("F1 Score",  f"{xgb_report['Phishing']['f1-score']:.2%}")
 
    # Random Forest metrics
    st.subheader("Random Forest")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Accuracy",  f"{accuracy_score(y_test, rf_pred):.2%}")
    r2.metric("Precision", f"{rf_report['Phishing']['precision']:.2%}")
    r3.metric("Recall",    f"{rf_report['Phishing']['recall']:.2%}")
    r4.metric("F1 Score",  f"{rf_report['Phishing']['f1-score']:.2%}")
 
    st.markdown("---")
 
    # confusion matrix — both side by side — same as notebook STEP 5
    st.subheader("Confusion Matrix — XGBoost vs Random Forest")
    col_a, col_b = st.columns(2)
 
    with col_a:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Legit", "Phishing"],
                    yticklabels=["Legit", "Phishing"], ax=ax)
        ax.set_title("Confusion Matrix — XGBoost")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()
 
    with col_b:
        cm_rf = confusion_matrix(y_test, rf_pred)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=["Legit", "Phishing"],
                    yticklabels=["Legit", "Phishing"], ax=ax2)
        ax2.set_title("Confusion Matrix — Random Forest")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)
        plt.close()
 
    st.markdown("---")
 
    # ROC curve — both — same as notebook STEP 6
    st.subheader("ROC Curve — XGBoost vs Random Forest")
    fpr,    tpr,    _ = roc_curve(y_test, y_pred_prob)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_prob)
    roc_auc    = auc(fpr,    tpr)
    rf_roc_auc = auc(rf_fpr, rf_tpr)
 
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(fpr,    tpr,    color="steelblue",  lw=2, label=f"XGBoost       (AUC = {roc_auc:.4f})")
    ax3.plot(rf_fpr, rf_tpr, color="darkorange", lw=2, label=f"Random Forest (AUC = {rf_roc_auc:.4f})")
    ax3.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve — XGBoost vs Random Forest")
    ax3.legend(loc="lower right")
    st.pyplot(fig3)
    plt.close()
 
    st.markdown("---")
 
    # precision recall — both — same as notebook STEP 7
    st.subheader("Precision-Recall Curve — XGBoost vs Random Forest")
    precision,    recall,    _ = precision_recall_curve(y_test, y_pred_prob)
    rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_pred_prob)
 
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.plot(recall,    precision,    color="steelblue",  lw=2, label="XGBoost")
    ax4.plot(rf_recall, rf_precision, color="darkorange", lw=2, label="Random Forest")
    ax4.set_xlabel("Recall")
    ax4.set_ylabel("Precision")
    ax4.set_title("Precision-Recall Curve — XGBoost vs Random Forest")
    ax4.legend(loc="lower left")
    st.pyplot(fig4)
    plt.close()
 
    st.markdown("---")
 
    # feature importance — both — same as notebook STEP 8
    st.subheader("Top 15 Feature Importances — XGBoost vs Random Forest")
    col_c, col_d = st.columns(2)
 
    with col_c:
        feat_df = pd.DataFrame({
            "Feature"    : shared_cols,
            "Importance" : model.feature_importances_
        }).sort_values("Importance", ascending=False).head(15)
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis", ax=ax5)
        ax5.set_title("Top 15 Features — XGBoost")
        ax5.set_xlabel("Importance Score")
        st.pyplot(fig5)
        plt.close()
 
    with col_d:
        rf_feat_df = pd.DataFrame({
            "Feature"    : shared_cols,
            "Importance" : rf_model.feature_importances_
        }).sort_values("Importance", ascending=False).head(15)
        fig6, ax6 = plt.subplots(figsize=(6, 5))
        sns.barplot(data=rf_feat_df, x="Importance", y="Feature", palette="magma", ax=ax6)
        ax6.set_title("Top 15 Features — Random Forest")
        ax6.set_xlabel("Importance Score")
        st.pyplot(fig6)
        plt.close()
 
    st.markdown("---")
 
    # confusion matrix after retraining — same as notebook STEP 12
    st.subheader("Confusion Matrix After Retraining — XGBoost vs Random Forest")
    col_e, col_f = st.columns(2)
 
    with col_e:
        cm2 = confusion_matrix(y_small, y_pred_new)
        fig7, ax7 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens",
                    xticklabels=["Legit", "Phishing"],
                    yticklabels=["Legit", "Phishing"], ax=ax7)
        ax7.set_title("Confusion Matrix After Retraining — XGBoost")
        ax7.set_xlabel("Predicted")
        ax7.set_ylabel("Actual")
        st.pyplot(fig7)
        plt.close()
 
    with col_f:
        cm3 = confusion_matrix(y_small, rf_pred_new)
        fig8, ax8 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm3, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=["Legit", "Phishing"],
                    yticklabels=["Legit", "Phishing"], ax=ax8)
        ax8.set_title("Confusion Matrix After Retraining — Random Forest")
        ax8.set_xlabel("Predicted")
        ax8.set_ylabel("Actual")
        st.pyplot(fig8)
        plt.close()
 
    st.markdown("---")
 
    # full classification reports
    st.subheader("Full Classification Reports")
    col_g, col_h = st.columns(2)
    with col_g:
        st.markdown("**XGBoost**")
        st.dataframe(pd.DataFrame(xgb_report).T.round(4), use_container_width=True)
    with col_h:
        st.markdown("**Random Forest**")
        st.dataframe(pd.DataFrame(rf_report).T.round(4), use_container_width=True)
 
 
# -------------------------------------------------------
# PAGE 3 — DRIFT ANALYSIS
# -------------------------------------------------------
 
elif "Drift" in page:
 
    st.title("🌊 Drift Analysis")
    st.markdown("KS Test comparing **dataset_full** (training) vs **dataset_small** (new data).")
    st.markdown("---")
 
    # same logic as notebook STEP 9
    drift_results    = {}
    drifted_features = []
 
    for col in shared_cols:
        stat, p_value = ks_2samp(X_full[col], X_small[col])
        drifted = p_value < 0.05
        drift_results[col] = {
            "KS Statistic" : round(stat,    4),
            "P-Value"      : round(p_value, 4),
            "Drifted"      : drifted
        }
        if drifted:
            drifted_features.append(col)
 
    drift_score = len(drifted_features) / len(shared_cols)
 
    c1, c2, c3 = st.columns(3)
    c1.metric("Drift Detected",   "Yes ⚠️" if drifted_features else "No ✅")
    c2.metric("Features Drifted", f"{len(drifted_features)} / {len(shared_cols)}")
    c3.metric("Drift Score",      f"{drift_score:.0%}")
 
    st.markdown("---")
    col_a, col_b = st.columns(2)
 
    # drift bar chart — same as notebook STEP 10
    with col_a:
        st.subheader("Top 20 drifted features (KS score)")
        drift_df = pd.DataFrame([
            {"Feature": col, "KS Statistic": v["KS Statistic"], "Drifted": v["Drifted"]}
            for col, v in drift_results.items()
        ]).sort_values("KS Statistic", ascending=False).head(20)
 
        colors = ["crimson" if d else "steelblue" for d in drift_df["Drifted"]]
        fig, ax = plt.subplots(figsize=(6, 7))
        ax.barh(drift_df["Feature"], drift_df["KS Statistic"], color=colors)
        ax.axvline(x=0.05, color="black", linestyle="--",
                   linewidth=1.2, label="Threshold p=0.05")
        ax.set_xlabel("KS Statistic")
        ax.set_title("Red = Drifted   |   Blue = Stable")
        ax.legend()
        st.pyplot(fig)
        plt.close()
 
    # feature distribution — same as notebook STEP 11
    with col_b:
        st.subheader("Feature distribution comparison")
        selected = st.selectbox(
            "Pick a feature to inspect:",
            options=drifted_features
        )
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(X_full[selected],  bins=40, alpha=0.6,
                 color="steelblue", label="Training — dataset_full")
        ax2.hist(X_small[selected], bins=40, alpha=0.6,
                 color="crimson",   label="New data — dataset_small")
        ax2.set_title(f"{selected}")
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Count")
        ax2.legend()
        st.pyplot(fig2)
        plt.close()
 
    st.markdown("---")
    st.subheader("Full drift report")
 
    drift_table = pd.DataFrame(drift_results).T.reset_index()
    drift_table.columns = ["Feature", "KS Statistic", "P-Value", "Drifted"]
    drift_table = drift_table.sort_values("KS Statistic", ascending=False)
 
    def highlight(row):
        bg = "background-color: #ffe5e5" if row["Drifted"] else "background-color: #e5ffe5"
        return [bg] * len(row)
 
    st.dataframe(
        drift_table.style.apply(highlight, axis=1),
        use_container_width=True,
        height=400
    )
 
 
# -------------------------------------------------------
# PAGE 4 — URL CHECKER
# -------------------------------------------------------
 
elif "URL" in page:
 
    st.title("🔗 URL Checker")
    st.markdown("Type any URL. Choose which model to use for prediction.")
    st.markdown("---")
 
    selected_model_name = st.radio(
        "Select model:",
        ["XGBoost", "Random Forest"],
        horizontal=True
    )
    selected_model   = model    if selected_model_name == "XGBoost" else rf_model
    other_model      = rf_model if selected_model_name == "XGBoost" else model
    other_model_name = "Random Forest" if selected_model_name == "XGBoost" else "XGBoost"
 
    with st.expander("Try these example URLs"):
        st.code("https://www.google.com/search?q=weather")
        st.code("http://secure-login-paypal-verify.xyz/account/confirm?token=abc123")
        st.code("https://www.amazon.co.uk/orders/history")
        st.code("http://free-prize-winner-click-now.login-verify.net/claim?id=99999")
        st.code("https://www.bbc.co.uk/news")
        st.code("http://192.168.1.1/login-secure-update-account-verify.php")
 
    url_input = st.text_input("Enter URL here:", placeholder="http://example.com/page")
 
    if st.button("🔍  Check this URL") and url_input:
 
        def extract_features(url):
            row = {col: 0 for col in shared_cols}
            mapping = {
                "qty_dot_url"          : url.count("."),
                "qty_hyphen_url"       : url.count("-"),
                "qty_underline_url"    : url.count("_"),
                "qty_slash_url"        : url.count("/"),
                "qty_questionmark_url" : url.count("?"),
                "qty_equal_url"        : url.count("="),
                "qty_at_url"           : url.count("@"),
                "qty_and_url"          : url.count("&"),
                "qty_exclamation_url"  : url.count("!"),
                "qty_space_url"        : url.count(" "),
                "qty_tilde_url"        : url.count("~"),
                "qty_comma_url"        : url.count(","),
                "qty_plus_url"         : url.count("+"),
                "qty_asterisk_url"     : url.count("*"),
                "qty_hashtag_url"      : url.count("#"),
                "qty_dollar_url"       : url.count("$"),
                "qty_percent_url"      : url.count("%"),
                "length_url"           : len(url),
                "email_in_url"         : int("@" in url),
                "url_shortened"        : int(any(s in url for s in ["bit.ly","tinyurl","goo.gl","t.co","ow.ly"])),
            }
            try:
                domain = url.split("/")[2] if "//" in url else url.split("/")[0]
                mapping["qty_dot_domain"]    = domain.count(".")
                mapping["qty_hyphen_domain"] = domain.count("-")
                mapping["domain_length"]     = len(domain)
                mapping["domain_in_ip"]      = int(domain.replace(".", "").isdigit())
                mapping["qty_vowels_domain"] = sum(1 for c in domain if c in "aeiou")
            except Exception:
                pass
            mapping["tls_ssl_certificate"] = int(url.startswith("https"))
            for key, val in mapping.items():
                if key in row:
                    row[key] = val
            return row
 
        features    = extract_features(url_input)
        X_url       = pd.DataFrame([features])
        prob        = selected_model.predict_proba(X_url)[0][1]
        label       = selected_model.predict(X_url)[0]
        other_prob  = other_model.predict_proba(X_url)[0][1]
        other_label = other_model.predict(X_url)[0]
 
        st.markdown("---")
 
        if label == 1:
            st.error(f"## ⚠️  PHISHING DETECTED  ({selected_model_name})")
            st.markdown(f"**Confidence: {prob:.2%}**  — This URL shows phishing characteristics.")
        else:
            st.success(f"## ✅  LEGITIMATE URL  ({selected_model_name})")
            st.markdown(f"**Confidence: {1 - prob:.2%}**  — This URL looks safe.")
 
        st.markdown("---")
 
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{selected_model_name} Probability", f"{prob:.2%}")
        c2.metric("Prediction", "Phishing ⚠️" if label == 1 else "Legit ✅")
        c3.metric("URL Length", len(url_input))
 
        st.markdown("#### Both model predictions")
        col_x, col_y = st.columns(2)
        with col_x:
            st.metric(f"{selected_model_name}",
                      "Phishing ⚠️" if label == 1 else "Legit ✅",
                      delta=f"{prob:.2%} phishing probability")
        with col_y:
            st.metric(f"{other_model_name}",
                      "Phishing ⚠️" if other_label == 1 else "Legit ✅",
                      delta=f"{other_prob:.2%} phishing probability")
 
        st.markdown("#### Prediction confidence — both models")
        fig, axes = plt.subplots(1, 2, figsize=(10, 1.5))
        for i, (mdl_name, mdl_prob) in enumerate(
            [(selected_model_name, prob), (other_model_name, other_prob)]
        ):
            axes[i].barh(["Legit", "Phishing"], [1 - mdl_prob, mdl_prob],
                         color=["steelblue", "crimson"])
            axes[i].set_xlim(0, 1)
            axes[i].axvline(x=0.5, color="black", linestyle="--", linewidth=1)
            axes[i].set_title(mdl_name)
            axes[i].set_xlabel("Probability")
            for spine in ["top", "right"]:
                axes[i].spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
 
        st.markdown("---")
        st.subheader("Features extracted from this URL")
        feat_show = {k: v for k, v in features.items() if v != 0}
        feat_df   = pd.DataFrame(feat_show.items(), columns=["Feature", "Value"])
        st.dataframe(feat_df, use_container_width=True, height=300)
 
        st.caption(
            "Note: The model was trained on 111 features including network-level signals "
            "(TTL, DNS, SSL, redirects). Those require live lookup tools. "
            "URL-string features are extracted here directly."
        )
 