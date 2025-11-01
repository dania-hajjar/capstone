# app.py
# From Feeds to Feelings: What the Data Says About How Youth Make Sense of Sexual Health
# RF page now mirrors the notebook block (median split, RandomizedSearchCV 5-fold, metrics, cls report, CM, ROC).

import io, re
import numpy as np
import pandas as pd
import streamlit as st

# Viz
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# ML
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import randint

# ---------------------------
# THEME / PALETTE
# ---------------------------
PALETTE = ["#2563eb", "#16a34a", "#f97316", "#ef4444", "#6b7280",
           "#0ea5e9", "#8b5cf6", "#f59e0b", "#10b981"]
PRIMARY, SECONDARY = PALETTE[0], PALETTE[1]
sns.set_theme(style="whitegrid")
sns.set_palette(PALETTE)
plt.rcParams.update({"axes.edgecolor":"#111","axes.labelcolor":"#111",
                     "xtick.color":"#111","ytick.color":"#111","text.color":"#111"})

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        font=dict(color="#111"),
        title_font=dict(color="#111"),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        xaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb", linecolor="#111"),
        yaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb", linecolor="#111"),
        colorway=PALETTE
    )
)

# ---------------------------
# UTILS
# ---------------------------
def find_col(df: pd.DataFrame, target: str):
    t = target.strip().lower()
    for c in df.columns:
        if c.strip().lower() == t:
            return c
    return None

def safe_to_datetime(series, errors="coerce"):
    try:
        return pd.to_datetime(series, errors=errors)
    except Exception:
        return pd.to_datetime(pd.Series([None]*len(series)), errors="coerce")

def robust_hashtag_count_from_text(text: str) -> int:
    if not isinstance(text, str): return 0
    text = re.sub(r"\s+", " ", text.strip())
    return len(re.findall(r"#\S+", text))

def ensure_week_parts(df, date_col):
    if date_col is None or date_col not in df.columns:
        df["weekday"] = pd.NA
        df["week_of_year"] = pd.NA
        return df
    dt = pd.to_datetime(df[date_col], errors="coerce")   # <-- fixed
    df["weekday"] = dt.dt.day_name()
    try:
        df["week_of_year"] = dt.dt.isocalendar().week.astype("Int64")
    except Exception:
        df["week_of_year"] = pd.NA
    return df

def extract_hour(df, date_col):
    for col in df.columns:
        if "time" in col.strip().lower():
            hh = safe_to_datetime(df[col]).dt.hour
            if hh.notna().any():
                df["hour"] = hh; return df
    if date_col and date_col in df.columns:
        dt = safe_to_datetime(df[date_col])
        if dt.dt.hour.notna().any():
            df["hour"] = dt.dt.hour; return df
    df["hour"] = np.nan; return df

def encode_topic_and_weekday(df):
    if "Topic" in df.columns and "Topic_encoded" not in df.columns:
        topic_mapping = {topic: i+1 for i, topic in enumerate(df["Topic"].astype(str).unique())}
        df["Topic_encoded"] = df["Topic"].map(topic_mapping)
    weekday_mapping = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
    if "weekday" in df.columns:
        df["weekday_encoded"] = df["weekday"].map(weekday_mapping)
    return df

def hashtag_groups(n):
    try:
        n = int(n)
    except Exception:
        return ">6"
    if n == 0: return "0"
    if 1 <= n <= 3: return "1–3"
    if 4 <= n <= 6: return "4–6"
    return ">6"

def caption_bins(s):
    if s is None: return None
    bins = [0,50,100,150,200,300,500, np.inf]
    labels = ["0–50","51–100","101–150","151–200","201–300","301–500",">500"]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True)

# --- File helpers ---
def load_file(uploaded):
    if uploaded is None: return None
    name = uploaded.name.lower()
    file_bytes = uploaded.getvalue()
    if name.endswith(".csv"):
        return {"kind":"csv","bytes":file_bytes,"sheets":None}
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        return {"kind":"excel","bytes":file_bytes,"sheets":xls.sheet_names}
    else:
        st.error("Please upload a CSV or Excel file."); return None

def read_dataframe_from_state(state, sheet_name=None):
    if state is None: return None
    if state["kind"] == "csv":
        return pd.read_csv(io.BytesIO(state["bytes"]))
    else:
        if sheet_name is None: sheet_name = state["sheets"][0]
        return pd.read_excel(io.BytesIO(state["bytes"]), sheet_name=sheet_name)

def merge_sent_topics(base_df: pd.DataFrame, sent_df: pd.DataFrame) -> pd.DataFrame:
    if base_df is None or sent_df is None: return base_df
    base_df.columns = [str(c) for c in base_df.columns]
    sent_df.columns = [str(c) for c in sent_df.columns]
    for key in ["Instagram URL", "Post title"]:
        if key in base_df.columns and key in sent_df.columns:
            newcols = [c for c in sent_df.columns if c not in base_df.columns]
            return base_df.merge(sent_df[[key]+newcols], on=key, how="left")
    if len(base_df) == len(sent_df):
        base = base_df.reset_index(drop=True).copy()
        add  = sent_df.reset_index(drop=True).copy()
        newcols = [c for c in add.columns if c not in base.columns]
        if newcols: base[newcols] = add[newcols]
        return base
    return base_df

# Cleaning pipeline
def run_cleaning_pipeline(raw_df: pd.DataFrame) -> dict:
    df = raw_df.copy()
    post_boosted_col = find_col(df, "Post Boosted")
    date_share_col   = find_col(df, "Date expected to share")
    hashtags_col     = find_col(df, "hashtags") or find_col(df, "hashtags ")
    description_col  = find_col(df, "Description")

    df["post_boosted"] = (
        df[post_boosted_col].astype(str).str.lower().str.contains("yes", na=False).astype(int)
        if post_boosted_col else 0
    )

    df = ensure_week_parts(df, date_share_col)
    df = extract_hour(df, date_share_col)

    acc_eng_col = find_col(df, "Accounts Engaged")
    acc_rea_col = find_col(df, "Accounts Reached")
    if acc_eng_col and acc_rea_col:
        engaged = pd.to_numeric(df[acc_eng_col], errors="coerce")
        reached = pd.to_numeric(df[acc_rea_col], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            df["engagement_rate"] = np.where(reached > 0, (engaged / reached) * 100, np.nan)
    elif "engagement_rate" in df.columns:
        df["engagement_rate"] = pd.to_numeric(df["engagement_rate"], errors="coerce")
    else:
        df["engagement_rate"] = np.nan

    df["caption_length"] = (
        df[description_col].astype(str).apply(len) if description_col else np.nan
    )

    if hashtags_col:
        df["n_hashtags"] = df[hashtags_col].astype(str).str.findall(r'#\w+').str.len()
        df["has_hashtags"] = (df["n_hashtags"] > 0).astype(int)
    else:
        src = df[description_col].astype(str) if description_col else pd.Series([""]*len(df))
        df["n_hashtags"] = src.apply(robust_hashtag_count_from_text)
        df["has_hashtags"] = (df["n_hashtags"] > 0).astype(int)

    df = encode_topic_and_weekday(df)

    preferred_order = [
        "Likes","Accounts Reached","Accounts Engaged","Impressions",
        "Post Interaction","post_boosted","week_of_year",
        "engagement_rate","caption_length","n_hashtags","has_hashtags",
        "Topic_encoded","weekday_encoded","hour"
    ]
    numerical_cols_for_imputation = [c for c in preferred_order if c in df.columns]
    if not numerical_cols_for_imputation:
        numerical_cols_for_imputation = df.select_dtypes(include=np.number).columns.tolist()

    before_na = df[numerical_cols_for_imputation].isna().sum()

    imputer = KNNImputer(n_neighbors=5)
    imputed_array = imputer.fit_transform(df[numerical_cols_for_imputation])
    df[numerical_cols_for_imputation] = pd.DataFrame(
        imputed_array, columns=numerical_cols_for_imputation, index=df.index
    )

    after_na = df[numerical_cols_for_imputation].isna().sum()

    df["Hashtag_Group"] = df["n_hashtags"].apply(hashtag_groups)
    df["Caption_Length_Group"] = caption_bins(df["caption_length"])
    if df["engagement_rate"].notna().any():
        thr = df["engagement_rate"].median(skipna=True)
        df["high_engagement"] = (df["engagement_rate"] > thr).astype(int)
    else:
        df["high_engagement"] = np.nan

    return {"df_clean": df, "impute_cols": numerical_cols_for_imputation,
            "missing_before": before_na, "missing_after": after_na}

# ---------------------------
# PAGE SETUP
# ---------------------------
st.set_page_config(page_title="Social Media Analytics — Capstone", page_icon="📊", layout="wide")
st.title("From Feeds to Feelings: What the Data Says About How Youth Make Sense of Sexual Health")

st.sidebar.title("📊 Capstone App")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
sent_topics_file = st.sidebar.file_uploader(
    "Optional: Sentiment/Topics CSV", type=["csv"],
    help="Upload the CSV from your notebook that includes sentiment_label/topic columns."
)

file_state = load_file(uploaded)
df_raw = None
if file_state is not None:
    if file_state["kind"] == "csv":
        df_raw = read_dataframe_from_state(file_state)
    else:
        sheets = file_state["sheets"]
        default_sheet = "Workplan " if "Workplan " in sheets else sheets[0]
        chosen_sheet = st.sidebar.selectbox("Excel sheet", sheets, index=sheets.index(default_sheet))
        df_raw = read_dataframe_from_state(file_state, sheet_name=chosen_sheet)

# Merge sentiment/topics (if provided) before cleaning
if df_raw is not None and sent_topics_file is not None:
    df_sent = pd.read_csv(sent_topics_file)
    df_raw = merge_sent_topics(df_raw, df_sent)

pages = st.sidebar.radio(
    "Navigate",
    ["Upload & Clean","Feature Engineering & Imputation","Visuals",
     "Models: Random Forest (Binary)","Clustering: K-Means (k=8)","Download Clean Data"],
    index=0
)

if df_raw is not None:
    pipeline_out = run_cleaning_pipeline(df_raw)
    df_clean = pipeline_out["df_clean"]
else:
    df_clean = None

# ---------------------------
# PAGE: Upload & Clean
# ---------------------------
if pages == "Upload & Clean":
    st.header("Upload & Raw Preview")
    if df_raw is None:
        st.info("Please upload your dataset and (if Excel) choose the **Workplan ** sheet.")
    else:
        st.write("**Raw head (first 20 rows):**")
        st.dataframe(df_raw.head(20), use_container_width=True)
        st.write("**Detected columns (raw):**")
        st.code(", ".join(df_raw.columns.astype(str)))

# ---------------------------
# PAGE: Feature Engineering & Imputation
# ---------------------------
elif pages == "Feature Engineering & Imputation":
    st.header("Cleaning, Feature Engineering & KNN Imputation")
    if df_clean is None:
        st.info("Upload a dataset first.")
    else:
        with st.expander("Steps replicated from the notebook", expanded=True):
            st.markdown("""
- `post_boosted` (Yes/No → 1/0)  
- `weekday`, `week_of_year`, `hour`  
- `engagement_rate` = Accounts Engaged / Accounts Reached × 100  
- `caption_length` & **Caption_Length_Group**  
- `n_hashtags`, `has_hashtags` & **Hashtag_Group**  
- `Topic_encoded` & `weekday_encoded`  
- **KNNImputer (k=5)** on numeric columns  
- `high_engagement` (median split)
""")
        st.subheader("Cleaned Data (head)")
        st.dataframe(df_clean.head(20), use_container_width=True)

        st.subheader("Imputation diagnostics (numeric set)")
        col1, col2 = st.columns(2)

        # use st.table (plain) to avoid React #185
        cols_used_df = pd.DataFrame({"imputed_numeric_columns": [str(c) for c in pipeline_out["impute_cols"]]})
        col1.write("**Columns used for KNN imputation:**")
        col1.table(cols_used_df)

        before_na = pipeline_out["missing_before"].copy()
        after_na  = pipeline_out["missing_after"].copy()
        before_na.index = before_na.index.astype(str)
        after_na.index  = after_na.index.astype(str)
        diag = (pd.concat([before_na.rename("Before"), after_na.rename("After")], axis=1)
                  .reset_index().rename(columns={"index":"Column"}))
        diag["Before"] = pd.to_numeric(diag["Before"]).fillna(0).astype(int)
        diag["After"]  = pd.to_numeric(diag["After"]).fillna(0).astype(int)
        col2.write("**Missing values (before → after):**")
        col2.table(diag)

# ---------------------------
# PAGE: Visuals
# ---------------------------
elif pages == "Visuals":
    st.header("Exploratory Visuals")
    if df_clean is None:
        st.info("Upload a dataset first.")
    else:
        if "sentiment_label" not in df_clean.columns:
            df_clean["sentiment_label"] = "Unknown"
        topic_col = None
        for c in ["topic_labels","Topic_label","Topic","Topic_encoded"]:
            if c in df_clean.columns:
                topic_col = c; break
        if topic_col is None:
            df_clean["Topic_fallback"] = "Unknown Topic"
            topic_col = "Topic_fallback"

        st.subheader("Average Engagement Rate by Weekday")
        if "weekday" in df_clean.columns and df_clean["engagement_rate"].notna().any():
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            tmp = (df_clean.groupby("weekday", dropna=False)["engagement_rate"]
                   .mean().reindex(order).reset_index())
            st.plotly_chart(px.bar(tmp, x="weekday", y="engagement_rate",
                                   template=PLOTLY_TEMPLATE,
                                   title="Average Engagement Rate by Weekday"),
                            use_container_width=True)

        st.subheader("Average Engagement Rate by Week of Year")
        if "week_of_year" in df_clean.columns and df_clean["engagement_rate"].notna().any():
            tmp = df_clean.groupby("week_of_year", dropna=True)["engagement_rate"].mean().reset_index()
            tmp = tmp.sort_values("week_of_year")
            st.plotly_chart(px.line(tmp, x="week_of_year", y="engagement_rate",
                                    template=PLOTLY_TEMPLATE,
                                    title="Average Engagement Rate by Week of Year"),
                            use_container_width=True)

        st.subheader("Engagement Rate Distribution — Boosted vs Not Boosted")
        if "post_boosted" in df_clean.columns and df_clean["engagement_rate"].notna().any():
            show_df = df_clean.assign(Boosted=df_clean["post_boosted"].map({1:"Boosted",0:"Not Boosted"}))
            fig = px.box(show_df, x="Boosted", y="engagement_rate",
                         template=PLOTLY_TEMPLATE,
                         title="Engagement Rate Distribution: Boosted vs Not Boosted",
                         color="Boosted", color_discrete_sequence=[PRIMARY, SECONDARY])
            fig.update_xaxes(title="")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average Engagement Rate by Hashtag Count Group")
        if "Hashtag_Group" in df_clean.columns and df_clean["engagement_rate"].notna().any():
            tmp = (df_clean.groupby("Hashtag_Group")["engagement_rate"]
                   .mean().reindex(["0","1–3","4–6",">6"]).reset_index())
            st.plotly_chart(px.bar(tmp, x="Hashtag_Group", y="engagement_rate",
                                   template=PLOTLY_TEMPLATE,
                                   title="Average Engagement by Hashtag Group"),
                            use_container_width=True)

        st.subheader("Engagement Rate Distribution by Caption Length Group")
        if "Caption_Length_Group" in df_clean.columns and df_clean["engagement_rate"].notna().any():
            st.plotly_chart(px.box(df_clean.dropna(subset=["Caption_Length_Group"]),
                                   x="Caption_Length_Group", y="engagement_rate",
                                   template=PLOTLY_TEMPLATE,
                                   title="Engagement Rate by Caption Length Group"),
                            use_container_width=True)

        # Avg Engagement by Sentiment (always renders)
        st.subheader("Average Engagement Rate by Sentiment")
        tmp = (df_clean.groupby("sentiment_label")["engagement_rate"]
               .mean().reset_index().sort_values("engagement_rate", ascending=False))
        st.plotly_chart(px.bar(tmp, x="sentiment_label", y="engagement_rate",
                               template=PLOTLY_TEMPLATE,
                               title="Average Engagement by Sentiment"),
                        use_container_width=True)

        # Sentiment distribution (always renders)
        st.subheader("Sentiment Distribution Across Posts")
        counts = df_clean["sentiment_label"].value_counts(dropna=False).rename_axis("sentiment").reset_index(name="count")
        st.plotly_chart(px.pie(counts, names="sentiment", values="count", hole=0.5,
                               template=PLOTLY_TEMPLATE,
                               title="Sentiment Distribution Across Posts"),
                        use_container_width=True)

        st.subheader("Correlation Heatmap of Numerical Variables")
        num_df = df_clean.select_dtypes(include="number")
        if not num_df.empty and num_df.shape[1] >= 2:
            corr = num_df.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(9,7))
            sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig, use_container_width=True)

        st.subheader("Interactive Scatter (choose X & Y) with Trendline")
        numeric_cols = sorted([c for c in df_clean.select_dtypes(include="number").columns if df_clean[c].notna().any()])
        if len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            default_x = "caption_length" if "caption_length" in numeric_cols else numeric_cols[0]
            default_y = "engagement_rate" if "engagement_rate" in numeric_cols else numeric_cols[1]
            x_sel = c1.selectbox("X axis", numeric_cols, index=numeric_cols.index(default_x) if default_x in numeric_cols else 0)
            y_sel = c2.selectbox("Y axis", numeric_cols, index=numeric_cols.index(default_y) if default_y in numeric_cols else 1)
            plot_df = df_clean[[x_sel, y_sel]].dropna()
            if not plot_df.empty:
                st.plotly_chart(px.scatter(plot_df, x=x_sel, y=y_sel,
                                           template=PLOTLY_TEMPLATE, trendline="ols",
                                           title=f"{y_sel} vs {x_sel} (with OLS trendline)"),
                                use_container_width=True)

        st.subheader("Average Engagement Rate by Topic and Sentiment")
        pivot = df_clean.pivot_table(index=(topic_col), columns="sentiment_label",
                                     values="engagement_rate", aggfunc="mean", dropna=False)
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", cbar_kws={'label':'Avg Engagement (%)'})
        ax.set_title("Engagement Rate by Topic and Sentiment")
        st.pyplot(fig, use_container_width=True)

# ---------------------------
# PAGE: Models — Random Forest (Binary)  (NOTEBOOK-ALIGNED)
# ---------------------------
elif pages == "Models: Random Forest (Binary)":
    st.header("Random Forest — Binary (Low vs High Engagement) — Notebook-Aligned")
    if df_clean is None:
        st.info("Upload a dataset first.")
    else:
        # === Notebook flow ===
        df = df_clean.dropna(subset=["engagement_rate"]).copy()
        if df.empty:
            st.error("No rows with engagement_rate available after dropping NaNs.")
        else:
            # Binary target via median split
            threshold = df["engagement_rate"].median()
            df["high_engagement"] = (df["engagement_rate"] > threshold).astype(int)

            # Features (numeric only) & Target
            X = df.drop(columns=["engagement_rate", "high_engagement"], errors="ignore") \
                  .select_dtypes(include=[np.number]).copy()
            y = df["high_engagement"].copy()

            X = X.fillna(0)

            # Train/Test split (stratified)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # RandomizedSearchCV param space (exactly as notebook)
            param_distributions = {
                "n_estimators": randint(200, 1000),
                "max_depth": randint(3, 40),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10),
                "max_features": ["sqrt", "log2", None],
                "bootstrap": [True, False]
            }

            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            random_search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_distributions,
                n_iter=50,
                cv=5,
                scoring="f1",
                random_state=42,
                n_jobs=-1,
                verbose=0,
            )

            # Fit on training set
            random_search.fit(X_train, y_train)
            best_rf = random_search.best_estimator_

            # Show best params and CV score
            st.subheader("Best CV Params & Score (5-fold, F1)")
            bp_col, bs_col = st.columns([3,1])
            with bp_col:
                st.code(random_search.best_params_)
            with bs_col:
                st.metric("Best CV F1", f"{random_search.best_score_:.4f}")

            # Train/Test metrics
            y_train_pred = best_rf.predict(X_train)
            y_test_pred  = best_rf.predict(X_test)
            y_test_proba = best_rf.predict_proba(X_test)[:, 1]

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc  = accuracy_score(y_test,  y_test_pred)
            train_f1  = f1_score(y_train, y_train_pred)
            test_f1   = f1_score(y_test,  y_test_pred)
            train_prec= precision_score(y_train, y_train_pred)
            test_prec = precision_score(y_test,  y_test_pred)
            train_rec = recall_score(y_train, y_train_pred)
            test_rec  = recall_score(y_test,  y_test_pred)
            test_auc  = roc_auc_score(y_test, y_test_proba)

            st.subheader("Random Forest Classifier Performance (Binary)")
            perf_df = pd.DataFrame({
                "Set": ["Train", "Test"],
                "Accuracy": [train_acc, test_acc],
                "F1": [train_f1, test_f1],
                "Precision": [train_prec, test_prec],
                "Recall": [train_rec, test_rec]
            })
            st.dataframe(perf_df.style.format(precision=3), use_container_width=True)
            st.caption(f"Test ROC–AUC: **{test_auc:.3f}**")

            # Classification Report (Test)
            st.subheader("Classification Report (Test)")
            report_text = classification_report(y_test, y_test_pred, digits=3, target_names=["Low","High"])
            st.code(report_text, language="text")


            # ROC Curve (AUC plot) — notebook-style
            st.subheader("ROC Curve — Random Forest (Low vs High)")
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            fig2, ax2 = plt.subplots(figsize=(4.8,4.0))
            ax2.plot(fpr, tpr, label=f"AUC = {test_auc:.3f}")
            ax2.plot([0, 1], [0, 1], "k--", label="Chance")
            ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve — Random Forest (Low vs High)")
            ax2.legend(loc="lower right")
            st.pyplot(fig2, use_container_width=False)

            # Top-10 Feature Importances
            st.subheader("Top 10 Feature Importances")
            fi = (pd.DataFrame({"Feature": X.columns,
                                "Importance": best_rf.feature_importances_})
                  .sort_values("Importance", ascending=False)
                  .head(10)
                  .reset_index(drop=True))
            st.dataframe(fi.style.format({"Importance":"{:.4f}"}), use_container_width=True)

# ---------------------------
# PAGE: Clustering — K-Means (k=8)
# ---------------------------
elif pages == "Clustering: K-Means (k=8)":
    st.header("K-Means on Engineered Features (k=8)")
    if df_clean is None:
        st.info("Upload a dataset first.")
    else:
        if "weekday" in df_clean.columns:
            map_wd = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
            w = df_clean["weekday"].astype(str).map(map_wd).fillna(3)
        else:
            w = pd.Series([3]*len(df_clean))
        weekday_sin = np.sin(2*np.pi*w/7.0); weekday_cos = np.cos(2*np.pi*w/7.0)

        hour = df_clean["hour"].fillna(12) if "hour" in df_clean.columns else pd.Series([12]*len(df_clean))
        hour_sin = np.sin(2*np.pi*hour/24.0); hour_cos = np.cos(2*np.pi*hour/24.0)

        Xp = pd.DataFrame({
            "n_hashtags": df_clean.get("n_hashtags", 0),
            "has_hashtags": df_clean.get("has_hashtags", 0),
            "post_boosted": df_clean.get("post_boosted", 0),
            "weekday_sin": weekday_sin, "weekday_cos": weekday_cos,
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "engagement_rate": df_clean.get("engagement_rate", 0)
        }).astype(float)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xp)

        k = 8
        km = KMeans(n_clusters=k, n_init=25, random_state=42)
        labels = km.fit_predict(Xs)
        df_k = df_clean.copy(); df_k["cluster_k8"] = labels

        try:
            sil = silhouette_score(Xs, labels)
            st.write(f"**Silhouette score (k=8):** `{sil:.3f}`")
        except Exception:
            st.write("**Silhouette score:** N/A")

        st.subheader("Cluster Profiles (means)")
        prof_cols = [c for c in ["engagement_rate","n_hashtags","has_hashtags","post_boosted"] if c in df_k.columns]
        profile = df_k.groupby("cluster_k8")[prof_cols].mean().round(2)
        st.dataframe(profile, use_container_width=True)

        st.subheader("Cluster Sizes")
        counts = df_k["cluster_k8"].value_counts().sort_index().rename_axis("cluster").reset_index(name="count")
        st.plotly_chart(px.bar(counts, x="cluster", y="count", template=PLOTLY_TEMPLATE, title="Cluster Sizes (k=8)"),
                        use_container_width=True)

        st.subheader("Preferred Weekday/Hour (decoded)")
        df_k["weekday_sin"] = weekday_sin; df_k["weekday_cos"] = weekday_cos
        df_k["hour_sin"] = hour_sin; df_k["hour_cos"] = hour_cos
        angle_w = np.arctan2(df_k.groupby("cluster_k8")["weekday_sin"].mean(),
                             df_k.groupby("cluster_k8")["weekday_cos"].mean())
        week_pref = ((angle_w % (2*np.pi)) * 7.0 / (2*np.pi))
        week_idx = np.round(week_pref).astype(int) % 7
        idx_to_day = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
        pref_weekday = pd.Series(week_idx).map(idx_to_day).rename("preferred_weekday")

        angle_h = np.arctan2(df_k.groupby("cluster_k8")["hour_sin"].mean(),
                             df_k.groupby("cluster_k8")["hour_cos"].mean())
        hour_pref = ((angle_h % (2*np.pi)) * 24.0 / (2*np.pi))
        pref_hour = pd.Series(np.round(hour_pref).astype(int)%24).rename("preferred_hour")

        pref_df = pd.concat([pref_weekday, pref_hour], axis=1); pref_df.index.name = "cluster_k8"
        st.dataframe(pref_df, use_container_width=True)

        st.subheader("Cluster × Sentiment Heatmap (% of posts)")
        if "sentiment_label" not in df_k.columns:
            df_k["sentiment_label"] = "Unknown"
        ct = (pd.crosstab(df_k["cluster_k8"], df_k["sentiment_label"], normalize="index") * 100).round(1)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(ct, cmap="Blues", annot=True, fmt=".1f", cbar_kws={'label': '% within cluster'})
        ax.set_title("Sentiment composition by cluster (k=8)")
        ax.set_xlabel("Sentiment"); ax.set_ylabel("Cluster")
        st.pyplot(fig, use_container_width=True)

        st.markdown("**Download clustered data (k=8):**")
        csv = df_k.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="workplan_clustered_k8.csv", mime="text/csv")

# ---------------------------
# PAGE: Download Clean Data
# ---------------------------
elif pages == "Download Clean Data":
    st.header("Download the Cleaned Dataset")
    if df_clean is None:
        st.info("Upload a dataset first.")
    else:
        st.dataframe(df_clean.head(20), use_container_width=True)
        csv = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="workplan_cleaned.csv", mime="text/csv")
