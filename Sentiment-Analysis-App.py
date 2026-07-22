import numpy as np
import os
import io
import json
import ast
import time
import random
import base64
import tempfile
import gzip
from datetime import datetime, timedelta, date
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pathlib import Path
from typing import Iterable, Any
from zoneinfo import ZoneInfo
import re
import requests
import pandas as pd
import streamlit as st
import altair as alt
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer, util
from google import genai
from google.cloud import storage

def get_gcs_client():
    return storage.Client()


def download_blob(blob_path: str, local_path: str, bucket_name = 'tulane-risk-data') -> str:
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    local_file = Path(local_path)
    local_file.parent.mkdir(parents = True, exist_ok=True)
    blob.download_to_filename(str(local_file))
    return str(local_file)

def blob_exists(blob_path: str, bucket_name: 'tulane-risk-data') -> bool:
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.exists()

def ensure_local_file(blob_path: str, local_path: str, bucket_name = 'tulane-risk-data', force: bool = False) -> str:
    p = Path(local_path)
    if force or not p.exists():
        download_blob(blob_path, local_path)
    return str(p)

def load_csv_gz_from_gcs(blob_path: str, local_path:str, bucket_name = 'tulane-risk-data', **read_csv_kwargs):
    ensure_local_file(blob_path, local_path, force = True)
    return pd.read_csv(local_path, compression = 'gzip', low_memory = False, **read_csv_kwargs)

def load_csv_from_gcs(blob_path: str, local_path: str, bucket_name='tulane-risk-data', **read_csv_kwargs):
    ensure_local_file(blob_path, local_path, bucket_name=bucket_name, force=True)
    return pd.read_csv(local_path, low_memory=False, **read_csv_kwargs)

def load_json_from_gcs(blob_path: str, local_path:str, bucket_name='tulane-risk-data'):
    ensure_local_file(blob_path, local_path, force = True)
    with open(local_path, 'r', encoding = 'utf-8') as f:
        return json.load(f)

def save_lifecycle_registry_to_gcs(
    lifecycle_df,
    blob_name,
    local_path,
):
    local_path = Path(local_path)

    local_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lifecycle_df.to_csv(
        local_path,
        index=False,
    )

    client = storage.Client()

    bucket = client.bucket(bucket_name)

    blob = bucket.blob(blob_name)

    blob.upload_from_filename(
        str(local_path),
        content_type="text/csv",
    )

def _github_token() -> str:
    # Secret Manager can add a trailing newline; strip it
    tok = os.getenv("GITHUB_TOKEN", "").strip()
    return tok

def _assert_github_ok(show=True):
    tok = _github_token()
    if not tok:
        if show: st.error("GITHUB_TOKEN is missing in Cloud Run environment.")
        return False
    try:
        r = requests.get(
            "https://api.github.com/user",
            headers={"Authorization": f"token {tok}", "Accept": "application/vnd.github+json"},
            timeout=15,
        )
        if r.status_code == 200:
            if show: st.caption(f"GitHub auth OK as **{r.json().get('login','?')}**")
            return True
        else:
            if show: st.error(f"GitHub auth failed ({r.status_code}): {r.text[:200]}")
            return False
    except Exception as e:
        if show: st.error(f"GitHub /user check failed: {e}")
        return False
tt = os.getenv('STREAMLIT_SECRETS')
if tt:
    try:
        APP_SECRETS = toml.loads(tt)
    except Exception:
        pass
try:
    APP_SECRETS = dict(st.secrets)
except Exception:
    APP_SECRETS = {}

def get_secrets(path, default = None):
    cur = APP_SECRETS
    for part in path.split('.'):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return default
        if cur is None:
            return default
    return cur


st.set_page_config(page_title="Tulane Risk Dashboard")
st.sidebar.title("Navigation")
st.sidebar.markdown("Select a tool:")
selection = st.sidebar.selectbox("Choose a tool:", ["Article Risk Review","External Risk Snapshot", "Unmatched Topic Analysis", "Risk/Event Detector"])

if "current_tab" not in st.session_state:
    st.session_state.current_tab = selection

# If switching tabs, clear session except the current tab
if st.session_state.current_tab != selection:
    keys_to_keep = {"current_tab"}
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    st.session_state.current_tab = selection

if selection == "External Risk Snapshot":
    OWNER = "ERSRisk"
    REPO = "Tulane-Sentiment-Analysis"
    TAG = "BERTopic_results"
    ASSET = "BERTopic_Streamlit.csv.gz"

    LIFECYCLE_BLOB = "agent/risk_lifecycle_registry.csv"
    LIFECYCLE_LOCAL = "pipeline/resources/risk_lifecycle_registry.csv"

    STATUS_LABELS = {
        "new": "Newly Identified",
        "active": "Active Concern",
        "candidate": "Under Review",
        "cooling": "Activity Declining",
        "expired": "Closed",
    }

    STATUS_EXPLANATIONS = {
        "new": "This development was recently promoted for leadership awareness.",
        "active": "Current evidence supports continued monitoring or action.",
        "candidate": "A relevant signal has been detected, but more evidence may be needed before escalation.",
        "cooling": "Recent supporting evidence has declined. The item will close if that continues.",
        "expired": "The development is no longer shown in the active view.",
    }

    STATUS_ORDER = {
        "new": 1,
        "active": 2,
        "candidate": 3,
        "cooling": 4,
        "expired": 5,
    }

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @st.cache_data(show_spinner=True, ttl=1800)
    def get_csv_from_release(owner, repo, tag, asset, usecols=None) -> pd.DataFrame:
        token = _github_token()
        if not token:
            raise RuntimeError("GITHUB_TOKEN missing (not injected or empty).")

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"token {token}",
        }

        release_response = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}",
            headers=headers,
            timeout=60,
        )

        if release_response.status_code != 200:
            raise RuntimeError(
                f"Release lookup {release_response.status_code}: "
                f"{release_response.text[:300]}"
            )

        release_json = release_response.json()
        asset_object = next(
            (
                item
                for item in release_json.get("assets", [])
                if item.get("name") == asset
            ),
            None,
        )

        if not asset_object:
            raise RuntimeError(
                f"Asset '{asset}' not found in release '{tag}'."
            )

        download_response = requests.get(
            asset_object["browser_download_url"],
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/octet-stream",
            },
            timeout=120,
        )

        if download_response.status_code != 200:
            raise RuntimeError(
                f"Asset download {download_response.status_code}: "
                f"{download_response.text[:300]}"
            )

        return pd.read_csv(
            io.BytesIO(download_response.content),
            compression="gzip",
            low_memory=False,
            dtype=str,
            usecols=usecols,
        )

    def clean_text(value, fallback=""):
        if value is None or pd.isna(value):
            return fallback

        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "nat"}:
            return fallback

        return text

    def safe_number(value, default=0.0):
        number = pd.to_numeric(value, errors="coerce")
        return default if pd.isna(number) else float(number)

    def format_date(value, short=False):
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            return "Not available"

        return parsed.strftime("%b %d, %Y" if short else "%B %d, %Y")

    def parse_article_sources(value):
        if isinstance(value, list):
            return value

        if value is None or pd.isna(value):
            return []

        text = str(value).strip()
        if not text:
            return []

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    def parse_boolean(value):
        if isinstance(value, bool):
            return value

        return str(value).strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    def priority_label(score):
        score = safe_number(score)

        if score >= 0.80:
            return "Critical"
        if score >= 0.65:
            return "High"
        if score >= 0.50:
            return "Moderate"
        return "Watch"

    def evidence_strength(row):
        source_count = int(safe_number(row.get("source_packet_count", 0)))
        relevance = clean_text(
            row.get("institutional_relevance"),
            "",
        ).lower()

        if relevance == "direct" and source_count >= 3:
            return "Strong"
        if relevance == "direct" or source_count >= 3:
            return "Moderate"
        return "Early"

    def development_type(row):
        origin_type = clean_text(
            row.get("origin_type", row.get("source_type", "")),
            "",
        ).lower()

        source_count = int(safe_number(row.get("source_packet_count", 0)))
        relevance = clean_text(
            row.get("institutional_relevance"),
            "",
        ).lower()
        actionability = clean_text(
            row.get("actionability"),
            "",
        ).lower()
        priority = safe_number(row.get("priority_score", 0))

        if origin_type in {"one_off", "one_off_article", "immediate_alert"}:
            return "Immediate Alert"

        if relevance == "direct" and actionability == "high" and source_count <= 1:
            return "Immediate Alert"

        if relevance == "direct" and priority >= 0.80 and source_count <= 1:
            return "Immediate Alert"

        return "Developing Pattern"

    def severity_bucket(value):
        score = safe_number(value)

        if score >= 4.0:
            return "Critical"
        if score >= 3.0:
            return "Elevated"
        if score >= 2.0:
            return "Monitor"
        return "Low"

    def action_label(score, trend, event_count):
        score = safe_number(score)
        trend = safe_number(trend)
        event_count = int(safe_number(event_count))

        if score >= 4 or (score >= 3 and trend > 0.3):
            return "Escalate"
        if score >= 2 or trend > 0.2 or event_count >= 3:
            return "Monitor"
        return "Watch"

    def clean_event_label(label, fallback_title):
        label = clean_text(label, "")
        if label:
            return label

        fallback = clean_text(fallback_title, "Unlabeled signal")
        return fallback[:140]

    def apply_risk_mapping(data, mapping):
        data = data.copy()
        mapping = mapping.copy()

        mapping["old_risk"] = (
            mapping["old_risk"].fillna("").astype(str).str.strip()
        )
        mapping["dashboard_risk"] = (
            mapping["dashboard_risk"].fillna("").astype(str).str.strip()
        )

        if "Predicted_Risks_new" not in data.columns:
            data["Predicted_Risks_new"] = "No Risk"

        data["Predicted_Risks_new"] = (
            data["Predicted_Risks_new"]
            .fillna("No Risk")
            .astype(str)
            .str.strip()
        )

        data = data.merge(
            mapping[["old_risk", "dashboard_risk"]].drop_duplicates(),
            left_on="Predicted_Risks_new",
            right_on="old_risk",
            how="left",
        )

        data["Dashboard_Risk"] = data["dashboard_risk"].where(
            data["dashboard_risk"].fillna("").astype(str).str.strip().ne(""),
            data["Predicted_Risks_new"],
        )

        return data.drop(
            columns=["old_risk", "dashboard_risk"],
            errors="ignore",
        )

    def update_risk_pin(
        lifecycle_df,
        canonical_event_id,
        should_pin,
        pinned_by="streamlit_user",
    ):
        updated = lifecycle_df.copy()

        mask = (
            updated["canonical_event_id"]
            .astype(str)
            .eq(str(canonical_event_id))
        )

        if not mask.any():
            raise KeyError(
                "The selected event was not found in the lifecycle registry."
            )

        now_utc = pd.Timestamp.now(tz="UTC")
        updated.loc[mask, "is_pinned"] = bool(should_pin)
        updated.loc[mask, "pinned_at"] = now_utc if should_pin else pd.NaT
        updated.loc[mask, "pinned_by"] = pinned_by if should_pin else ""

        if should_pin:
            updated.loc[mask, "lifecycle_status"] = "active"
            updated.loc[mask, "expired_at"] = pd.NaT

        return updated

    def prepare_agent_decisions(data):
        data = data.copy()

        if data.empty:
            return data

        if "evaluation_timestamp" in data.columns:
            data["evaluation_timestamp"] = pd.to_datetime(
                data["evaluation_timestamp"],
                errors="coerce",
                utc=True,
            )

        if "validation_status" in data.columns:
            data = data[
                data["validation_status"]
                .fillna("valid")
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("valid")
            ].copy()

        relevance_ok = pd.Series(True, index=data.index)
        visibility_ok = pd.Series(True, index=data.index)
        review_ok = pd.Series(True, index=data.index)

        if "institutional_relevance" in data.columns:
            relevance_ok = (
                data["institutional_relevance"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("direct")
            )

        if "dashboard_visibility" in data.columns:
            visibility_ok = (
                data["dashboard_visibility"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("show")
            )

        if "requires_human_review" in data.columns:
            review_ok = ~data["requires_human_review"].apply(parse_boolean)

        data = data[relevance_ok & visibility_ok & review_ok].copy()

        if data.empty:
            return data

        dedup_key = (
            "canonical_event_id"
            if "canonical_event_id" in data.columns
            else "unit_id"
        )

        if "evaluation_timestamp" in data.columns:
            data = data.sort_values(
                "evaluation_timestamp",
                ascending=False,
                na_position="last",
            )

        return data.drop_duplicates(
            subset=[dedup_key],
            keep="first",
        )

    def render_article_sources(article_sources):
        if not article_sources:
            st.caption("No supporting articles are currently available.")
            return

        st.markdown("**Supporting evidence**")

        for article_number, article in enumerate(article_sources, start=1):
            title = clean_text(
                article.get("title"),
                f"Source {article_number}",
            )
            link = clean_text(article.get("link"), "")
            source = clean_text(
                article.get("source", article.get("publisher", "")),
                "",
            )
            published = format_date(article.get("published"), short=True)

            if link:
                st.markdown(f"**[{title}]({link})**")
            else:
                st.markdown(f"**{title}**")

            metadata = [
                item
                for item in [source, published]
                if item and item != "Not available"
            ]
            if metadata:
                st.caption(" · ".join(metadata))

            snippet = clean_text(article.get("snippet"), "")
            if snippet:
                with st.expander("View article summary", expanded=False):
                    st.write(snippet)

            if article_number < len(article_sources):
                st.divider()

    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------
    articles = load_csv_gz_from_gcs(
        "latest/topics/BERTopic_Streamlit.csv.gz",
        "pipeline/resources/BERTopic_Streamlit.csv.gz",
    )

    risk_scores = load_csv_from_gcs(
        "latest/scores/final_risk_scores1.csv",
        "pipeline/resources/final_risk_scores1.csv",
    )

    events = load_csv_from_gcs(
        "latest/scores/ranked_events_risks1.csv",
        "pipeline/resources/ranked_events_risks1.csv",
    )

    agent_decisions = load_csv_from_gcs(
        "agent/agent_decisions.csv",
        "pipeline/resources/agent_decisions.csv",
    )

    risk_lifecycle = load_csv_from_gcs(
        "agent/risk_lifecycle_registry.csv",
        "pipeline/resources/risk_lifecycle_registry.csv",
    )

    consolidated_packets = load_csv_from_gcs(
        "agent/consolidated_agent_packets.csv",
        "pipeline/resources/consolidated_agent_packets.csv",
    )

    emerging_situations = load_csv_from_gcs(
        "agent/emerging_situations.csv",
        "pipeline/resources/emerging_situations.csv",
    )

    risk_mapping = load_csv_from_gcs(
        "latest/reference/risk_mapping.csv",
        "pipeline/resources/risk_mapping.csv",
    )

    lifecycle_date_columns = [
        "first_seen",
        "last_seen",
        "last_evidence_at",
        "first_promoted_at",
        "last_qualified_at",
        "pinned_at",
        "expired_at",
    ]

    for column in lifecycle_date_columns:
        if column in risk_lifecycle.columns:
            risk_lifecycle[column] = pd.to_datetime(
                risk_lifecycle[column],
                errors="coerce",
                utc=True,
            )

    if "is_pinned" not in risk_lifecycle.columns:
        risk_lifecycle["is_pinned"] = False
    else:
        risk_lifecycle["is_pinned"] = risk_lifecycle["is_pinned"].apply(
            parse_boolean
        )

    agent_decisions = prepare_agent_decisions(agent_decisions)

    decision_columns = [
        "canonical_event_id",
        "agent_decision",
        "dashboard_visibility",
        "institutional_relevance",
        "actionability",
        "confidence",
        "what_changed",
        "why_it_matters",
        "policy_mitigation_assessment",
        "recommended_human_action",
        "relevant_policies",
        "coverage_gaps",
        "evaluation_timestamp",
        "requires_human_review",
    ]

    packet_columns = [
        "canonical_event_id",
        "top_articles",
        "top_titles",
        "source_packet_count",
    ]

    agent_subset = agent_decisions.reindex(columns=decision_columns)
    packet_subset = consolidated_packets.reindex(columns=packet_columns)

    emerging_risk_items = (
        risk_lifecycle
        .merge(
            agent_subset,
            on="canonical_event_id",
            how="left",
        )
        .merge(
            packet_subset,
            on="canonical_event_id",
            how="left",
        )
    )

    if "top_articles" not in emerging_risk_items.columns:
        emerging_risk_items["top_articles"] = None

    emerging_risk_items["article_sources"] = (
        emerging_risk_items["top_articles"].apply(parse_article_sources)
    )

    if "source_packet_count" not in emerging_risk_items.columns:
        emerging_risk_items["source_packet_count"] = (
            emerging_risk_items["article_sources"].apply(len)
        )
    else:
        emerging_risk_items["source_packet_count"] = pd.to_numeric(
            emerging_risk_items["source_packet_count"],
            errors="coerce",
        ).fillna(
            emerging_risk_items["article_sources"].apply(len)
        )

    valid_lifecycle_statuses = [
        "new",
        "active",
        "candidate",
        "cooling",
    ]

    emerging_risk_items = emerging_risk_items[
        emerging_risk_items["is_pinned"]
        | emerging_risk_items["lifecycle_status"].isin(valid_lifecycle_statuses)
    ].copy()

    emerging_risk_items["_status_order"] = (
        emerging_risk_items["lifecycle_status"]
        .map(STATUS_ORDER)
        .fillna(99)
    )
    emerging_risk_items["Priority"] = emerging_risk_items[
        "priority_score"
    ].apply(priority_label)
    emerging_risk_items["Status"] = emerging_risk_items[
        "lifecycle_status"
    ].map(STATUS_LABELS).fillna("Under Review")
    emerging_risk_items["Evidence Strength"] = emerging_risk_items.apply(
        evidence_strength,
        axis=1,
    )
    emerging_risk_items["Development Type"] = emerging_risk_items.apply(
        development_type,
        axis=1,
    )

    # Risk mapping and date preparation.
    articles = apply_risk_mapping(articles, risk_mapping)
    events = apply_risk_mapping(events, risk_mapping)
    risk_scores = apply_risk_mapping(risk_scores, risk_mapping)

    events["Window"] = pd.to_datetime(
        events.get("Window"),
        errors="coerce",
        utc=True,
    )
    risk_scores["Window"] = pd.to_datetime(
        risk_scores.get("Window"),
        errors="coerce",
        utc=True,
    )

    if "Published_utc" in articles.columns:
        articles["Published_utc"] = pd.to_datetime(
            articles["Published_utc"],
            errors="coerce",
            utc=True,
        )
    elif "Published" in articles.columns:
        articles["Published_utc"] = pd.to_datetime(
            articles["Published"],
            errors="coerce",
            utc=True,
        )
    else:
        articles["Published_utc"] = pd.NaT

    period_days = 30
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=period_days)
    previous_cutoff = cutoff - pd.Timedelta(days=period_days)

    current_events = events[events["Window"] >= cutoff].copy()
    previous_events = events[
        (events["Window"] >= previous_cutoff)
        & (events["Window"] < cutoff)
    ].copy()

    events["Event_Severity"] = pd.to_numeric(
        events.get("Event_Severity"),
        errors="coerce",
    ).fillna(0.0)
    current_events["Event_Severity"] = pd.to_numeric(
        current_events.get("Event_Severity"),
        errors="coerce",
    ).fillna(0.0)
    previous_events["Event_Severity"] = pd.to_numeric(
        previous_events.get("Event_Severity"),
        errors="coerce",
    ).fillna(0.0)

    risk_universe = (
        risk_mapping[["dashboard_risk"]]
        .dropna()
        .assign(
            dashboard_risk=lambda frame: (
                frame["dashboard_risk"].astype(str).str.strip()
            )
        )
        .loc[lambda frame: frame["dashboard_risk"].ne("")]
        .drop_duplicates()
        .rename(columns={"dashboard_risk": "Dashboard_Risk"})
    )

    current_scores = (
        current_events.groupby("Dashboard_Risk")["Event_Severity"]
        .mean()
        .reset_index(name="current_score")
    )

    previous_scores = (
        previous_events.groupby("Dashboard_Risk")["Event_Severity"]
        .mean()
        .reset_index(name="previous_score")
    )

    event_counts = (
        current_events.groupby("Dashboard_Risk")
        .size()
        .reset_index(name="Event_Count")
    )

    snapshot = (
        risk_universe
        .merge(current_scores, on="Dashboard_Risk", how="left")
        .merge(previous_scores, on="Dashboard_Risk", how="left")
        .merge(event_counts, on="Dashboard_Risk", how="left")
    )

    for column in ["current_score", "previous_score"]:
        snapshot[column] = pd.to_numeric(
            snapshot[column],
            errors="coerce",
        ).fillna(0.0)

    snapshot["Event_Count"] = pd.to_numeric(
        snapshot["Event_Count"],
        errors="coerce",
    ).fillna(0).astype(int)

    current_names = set(current_scores["Dashboard_Risk"].dropna().astype(str))
    previous_names = set(previous_scores["Dashboard_Risk"].dropna().astype(str))

    snapshot["is_new"] = (
        snapshot["Dashboard_Risk"].isin(current_names)
        & ~snapshot["Dashboard_Risk"].isin(previous_names)
    )
    snapshot["trend"] = snapshot["current_score"] - snapshot["previous_score"]
    snapshot.loc[snapshot["is_new"], "trend"] = snapshot.loc[
        snapshot["is_new"],
        "current_score",
    ]
    snapshot["severity_band"] = snapshot["current_score"].apply(
        severity_bucket
    )
    snapshot["action"] = snapshot.apply(
        lambda row: action_label(
            row["current_score"],
            row["trend"],
            row["Event_Count"],
        ),
        axis=1,
    )

    total_current_score = snapshot["current_score"].sum()
    snapshot["risk_share_pct"] = (
        snapshot["current_score"] / total_current_score * 100
        if total_current_score > 0
        else 0.0
    )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def render_executive_overview(snapshot_data, development_data):
        actionable = development_data[
            development_data["actionability"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("high")
        ]

        new_count = int(
            development_data["lifecycle_status"].eq("new").sum()
        )
        active_count = int(
            development_data["lifecycle_status"].eq("active").sum()
        )
        immediate_alert_count = int(
            development_data["Development Type"]
            .eq("Immediate Alert")
            .sum()
        )

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Risks Requiring Action", len(actionable))
        metric_2.metric("New Developments", new_count)
        metric_3.metric("Active Concerns", active_count)
        metric_4.metric("Immediate Alerts", immediate_alert_count)

        st.divider()

        left_column, right_column = st.columns([3, 2])

        with left_column:
            st.subheader("Current Risk Position")

            top_risks = (
                snapshot_data
                .sort_values(
                    ["current_score", "trend"],
                    ascending=[False, False],
                )
                .head(8)
                .copy()
            )

            top_risks["Current Level"] = top_risks["current_score"].round(2)
            top_risks["Change"] = top_risks["trend"].round(2)
            top_risks["Direction"] = top_risks["trend"].apply(
                lambda value: (
                    "Increasing"
                    if value > 0.2
                    else "Decreasing"
                    if value < -0.2
                    else "Stable"
                )
            )

            st.dataframe(
                top_risks[
                    [
                        "Dashboard_Risk",
                        "Current Level",
                        "Direction",
                        "Change",
                        "action",
                        "Event_Count",
                    ]
                ].rename(
                    columns={
                        "Dashboard_Risk": "Risk Area",
                        "action": "Recommended Posture",
                        "Event_Count": "Events",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

        with right_column:
            st.subheader("Leadership Priorities")

            leadership_items = (
                development_data
                .sort_values(
                    ["is_pinned", "priority_score", "last_evidence_at"],
                    ascending=[False, False, False],
                    na_position="last",
                )
                .head(5)
            )

            if leadership_items.empty:
                st.caption("No current leadership priorities were identified.")
            else:
                for rank, (_, row) in enumerate(
                    leadership_items.iterrows(),
                    start=1,
                ):
                    title = clean_text(
                        row.get("situation_title"),
                        "Untitled development",
                    )
                    risk_area = clean_text(
                        row.get("top_risk_match"),
                        "Unmapped risk",
                    )
                    action = clean_text(
                        row.get("recommended_human_action"),
                        "Continue monitoring and assess institutional exposure.",
                    )

                    st.markdown(f"**{rank}. {title}**")
                    st.caption(
                        f"{risk_area} · {row.get('Priority', 'Watch')} priority"
                    )
                    st.write(action)

                    if rank < len(leadership_items):
                        st.divider()

        st.divider()
        st.subheader("Fastest-Moving Risk Areas")

        trending = (
            snapshot_data
            .sort_values("trend", ascending=False)
            .head(5)
        )

        trend_columns = st.columns(5)
        for index, (_, row) in enumerate(trending.iterrows()):
            with trend_columns[index]:
                st.metric(
                    row["Dashboard_Risk"],
                    f"{row['current_score']:.2f}",
                    delta=f"{row['trend']:+.2f}",
                    delta_color="inverse",
                )

    def render_development_card(row):
        canonical_id = str(row.get("canonical_event_id", ""))
        title = clean_text(
            row.get("situation_title"),
            "Untitled risk development",
        )
        risk_area = clean_text(
            row.get("top_risk_match"),
            "Unmapped risk",
        )
        lifecycle_status = clean_text(
            row.get("lifecycle_status"),
            "candidate",
        ).lower()
        status_label = STATUS_LABELS.get(
            lifecycle_status,
            lifecycle_status.title(),
        )
        is_pinned = parse_boolean(row.get("is_pinned", False))
        article_sources = row.get("article_sources", [])
        article_sources = article_sources if isinstance(article_sources, list) else []
        priority = priority_label(row.get("priority_score"))
        evidence_label = evidence_strength(row)
        development_label = development_type(row)

        with st.container(border=True):
            heading_column, priority_column = st.columns([4, 1])

            with heading_column:
                st.markdown(f"### {title}")
                metadata = [
                    risk_area,
                    status_label,
                    development_label,
                    f"{len(article_sources)} source(s)",
                ]
                if is_pinned:
                    metadata.insert(0, "Leadership Watchlist")
                st.caption(" · ".join(metadata))

            with priority_column:
                st.markdown(f"**{priority} Priority**")
                st.caption(f"Evidence: {evidence_label}")

            why_it_matters = clean_text(row.get("why_it_matters"), "")
            if why_it_matters:
                st.write(why_it_matters)

            recommended_action = clean_text(
                row.get("recommended_human_action"),
                "",
            )
            if recommended_action:
                st.info(f"Recommended response: {recommended_action}")

            detail_column, watchlist_column = st.columns([4, 1])

            with detail_column:
                with st.expander(
                    "View evidence and lifecycle details",
                    expanded=False,
                ):
                    detail_1, detail_2, detail_3, detail_4 = st.columns(4)
                    detail_1.metric(
                        "First Observed",
                        format_date(row.get("first_seen"), short=True),
                    )
                    detail_2.metric(
                        "Latest Evidence",
                        format_date(row.get("last_evidence_at"), short=True),
                    )
                    detail_3.metric(
                        "Supporting Sources",
                        len(article_sources),
                    )
                    detail_4.metric(
                        "Model Score",
                        f"{safe_number(row.get('priority_score')):.2f}",
                    )

                    what_changed = clean_text(row.get("what_changed"), "")
                    if what_changed:
                        st.markdown("**What changed**")
                        st.write(what_changed)

                    policy_assessment = clean_text(
                        row.get("policy_mitigation_assessment"),
                        "",
                    )
                    if policy_assessment:
                        st.markdown("**Policy or mitigation context**")
                        st.write(policy_assessment)

                    status_explanation = STATUS_EXPLANATIONS.get(
                        lifecycle_status,
                        "This item will be reassessed as new evidence becomes available.",
                    )
                    st.caption(status_explanation)

                    render_article_sources(article_sources)

            with watchlist_column:
                button_text = (
                    "Remove from Watchlist"
                    if is_pinned
                    else "Add to Watchlist"
                )

                if st.button(
                    button_text,
                    key=f"watchlist_{canonical_id}",
                    use_container_width=True,
                ):
                    try:
                        updated_lifecycle = update_risk_pin(
                            lifecycle_df=risk_lifecycle,
                            canonical_event_id=canonical_id,
                            should_pin=not is_pinned,
                        )

                        save_lifecycle_registry_to_gcs(
                            lifecycle_df=updated_lifecycle,
                            blob_name=LIFECYCLE_BLOB,
                            local_path=LIFECYCLE_LOCAL,
                        )

                        st.cache_data.clear()
                        st.success(
                            "Development added to the leadership watchlist."
                            if not is_pinned
                            else "Development removed from the leadership watchlist."
                        )
                        st.rerun()

                    except Exception as error:
                        st.error(
                            "The lifecycle registry could not be updated: "
                            f"{error}"
                        )

    def render_developments(development_data):
        st.subheader("Developments Requiring Attention")
        st.caption(
            "New and evolving external developments that may require "
            "monitoring, assessment, or action."
        )

        filter_1, filter_2, filter_3 = st.columns(3)

        display_to_internal = {value: key for key, value in STATUS_LABELS.items()}
        available_status_labels = [
            STATUS_LABELS[status]
            for status in ["new", "active", "candidate", "cooling"]
        ]

        with filter_1:
            selected_status_labels = st.multiselect(
                "Status",
                available_status_labels,
                default=[
                    "Newly Identified",
                    "Active Concern",
                    "Under Review",
                ],
            )

        with filter_2:
            selected_priorities = st.multiselect(
                "Leadership Priority",
                ["Critical", "High", "Moderate", "Watch"],
                default=["Critical", "High", "Moderate"],
            )

        with filter_3:
            sort_choice = st.selectbox(
                "Sort by",
                [
                    "Highest priority",
                    "Most recent evidence",
                    "Newest development",
                    "Status",
                ],
            )

        selected_internal_statuses = [
            display_to_internal[label]
            for label in selected_status_labels
            if label in display_to_internal
        ]

        display_items = development_data[
            development_data["lifecycle_status"].isin(selected_internal_statuses)
            & development_data["Priority"].isin(selected_priorities)
        ].copy()

        if sort_choice == "Highest priority":
            display_items = display_items.sort_values(
                ["is_pinned", "priority_score", "last_evidence_at"],
                ascending=[False, False, False],
                na_position="last",
            )
        elif sort_choice == "Most recent evidence":
            display_items = display_items.sort_values(
                ["is_pinned", "last_evidence_at", "priority_score"],
                ascending=[False, False, False],
                na_position="last",
            )
        elif sort_choice == "Newest development":
            display_items = display_items.sort_values(
                ["is_pinned", "first_promoted_at", "priority_score"],
                ascending=[False, False, False],
                na_position="last",
            )
        else:
            display_items = display_items.sort_values(
                ["is_pinned", "_status_order", "priority_score"],
                ascending=[False, True, False],
                na_position="last",
            )

        immediate_alerts = display_items[
            display_items["Development Type"].eq("Immediate Alert")
        ]
        developing_patterns = display_items[
            display_items["Development Type"].eq("Developing Pattern")
        ]
        watchlist = display_items[display_items["is_pinned"]]

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Immediate Alerts", len(immediate_alerts))
        metric_2.metric("Developing Patterns", len(developing_patterns))
        metric_3.metric(
            "Newly Identified",
            int(display_items["lifecycle_status"].eq("new").sum()),
        )
        metric_4.metric("Leadership Watchlist", len(watchlist))

        st.divider()

        if display_items.empty:
            st.info("No developments match the selected filters.")
            return

        summary_table = display_items.copy()
        summary_table["Latest Evidence"] = summary_table[
            "last_evidence_at"
        ].apply(lambda value: format_date(value, short=True))
        summary_table["Development"] = summary_table["situation_title"].apply(
            lambda value: clean_text(value, "Untitled development")
        )
        summary_table["Risk Area"] = summary_table["top_risk_match"].apply(
            lambda value: clean_text(value, "Unmapped risk")
        )
        summary_table["Sources"] = summary_table["article_sources"].apply(
            lambda value: len(value) if isinstance(value, list) else 0
        )

        st.markdown("#### Development Summary")
        st.dataframe(
            summary_table[
                [
                    "Development",
                    "Risk Area",
                    "Development Type",
                    "Status",
                    "Priority",
                    "Evidence Strength",
                    "Sources",
                    "Latest Evidence",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

        st.divider()

        if not immediate_alerts.empty:
            st.markdown("## Immediate Alerts")
            st.caption(
                "High-relevance one-off developments that should not wait "
                "for recurrence before receiving attention."
            )
            for _, row in immediate_alerts.iterrows():
                render_development_card(row)

        if not developing_patterns.empty:
            st.markdown("## Developing Patterns")
            st.caption(
                "Recurring or multi-source developments whose evidence is "
                "building over time."
            )
            for _, row in developing_patterns.iterrows():
                render_development_card(row)

    def render_risk_explorer(snapshot_data, event_data, article_data):
        st.subheader("Risk Explorer")
        st.caption(
            "Select a risk category to examine its score, drivers, events, "
            "and supporting evidence."
        )

        all_risks = sorted(
            risk_mapping["dashboard_risk"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda series: series.ne("")]
            .unique(),
            key=str.lower,
        )

        if not all_risks:
            st.warning("No risk categories are available.")
            return

        filter_1, filter_2, filter_3, filter_4 = st.columns([2, 2, 1, 1])

        with filter_1:
            selected_risk = st.selectbox(
                "Risk Category",
                all_risks,
            )

        subcategory_options = sorted(
            risk_mapping.loc[
                risk_mapping["dashboard_risk"]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq(selected_risk),
                "old_risk",
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda series: series.ne("")]
            .unique(),
            key=str.lower,
        )

        with filter_2:
            selected_subcategory = st.selectbox(
                "Risk Subcategory",
                ["All Subcategories"] + subcategory_options,
            )

        with filter_3:
            minimum_severity = st.slider(
                "Minimum Severity",
                0.0,
                5.0,
                2.5,
                0.1,
            )

        with filter_4:
            maximum_items = st.selectbox(
                "Maximum Items",
                [10, 25, 50, 100],
                index=1,
            )

        search_text = st.text_input(
            "Search article headlines",
            "",
        )

        selected_snapshot = snapshot_data[
            snapshot_data["Dashboard_Risk"].eq(selected_risk)
        ]

        if not selected_snapshot.empty:
            selected_row = selected_snapshot.iloc[0]
            metric_1, metric_2, metric_3, metric_4 = st.columns(4)
            metric_1.metric(
                "Current Risk Level",
                f"{selected_row['current_score']:.2f}",
            )
            metric_2.metric(
                "Change vs. Prior Period",
                f"{selected_row['trend']:+.2f}",
            )
            metric_3.metric(
                "Recent Events",
                int(selected_row["Event_Count"]),
            )
            metric_4.metric(
                "Recommended Posture",
                selected_row["action"],
            )

            trend_word = (
                "increased"
                if selected_row["trend"] > 0
                else "decreased"
                if selected_row["trend"] < 0
                else "remained stable"
            )

            st.info(
                f"{selected_risk} is currently rated "
                f"**{selected_row['severity_band']}**. Its score {trend_word} "
                f"by **{abs(selected_row['trend']):.2f}** versus the prior "
                f"30-day period, based on **{int(selected_row['Event_Count'])}** "
                "recent event(s)."
            )

        risk_events = event_data[
            event_data["Dashboard_Risk"].eq(selected_risk)
            & (event_data["Window"] >= cutoff)
        ].copy()

        if selected_subcategory != "All Subcategories":
            risk_events = risk_events[
                risk_events["Predicted_Risks_new"]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq(selected_subcategory)
            ].copy()

        risk_events["Event_Severity"] = pd.to_numeric(
            risk_events.get("Event_Severity"),
            errors="coerce",
        ).fillna(0.0)
        risk_events = risk_events[
            risk_events["Event_Severity"] >= minimum_severity
        ].copy()

        # Merge article details by Link first, then Title.
        for frame in [risk_events, article_data]:
            if "Link" in frame.columns:
                frame["Link"] = frame["Link"].fillna("").astype(str).str.strip()
            if "Title" in frame.columns:
                frame["Title"] = frame["Title"].fillna("").astype(str).str.strip()

        article_detail_columns = [
            column
            for column in [
                "Event_Label",
                "Title",
                "Content",
                "Published_utc",
                "Link",
                "Source",
                "source",
                "canonical_source",
            ]
            if column in article_data.columns
        ]

        if (
            "Link" in risk_events.columns
            and "Link" in article_data.columns
            and risk_events["Link"].str.startswith("http").any()
        ):
            join_keys = ["Link"]
        elif "Title" in risk_events.columns and "Title" in article_data.columns:
            join_keys = ["Title"]
        else:
            join_keys = ["Event_Label"]

        if all(key in article_data.columns for key in join_keys):
            article_details = (
                article_data[article_detail_columns]
                .drop_duplicates(subset=join_keys)
            )
            risk_events = risk_events.merge(
                article_details,
                on=join_keys,
                how="left",
                suffixes=("", "_article"),
            )

            for column in [
                "Title",
                "Content",
                "Published_utc",
                "Link",
                "Source",
                "source",
                "canonical_source",
            ]:
                article_column = f"{column}_article"
                if article_column in risk_events.columns:
                    if column in risk_events.columns:
                        current_value_present = (
                            risk_events[column].notna()
                            & risk_events[column]
                            .astype(str)
                            .str.strip()
                            .ne("")
                        )
                        risk_events[column] = risk_events[column].where(
                            current_value_present,
                            risk_events[article_column],
                        )
                    else:
                        risk_events[column] = risk_events[article_column]

            risk_events = risk_events.drop(
                columns=[
                    column
                    for column in risk_events.columns
                    if column.endswith("_article")
                ],
                errors="ignore",
            )

        if search_text.strip() and "Title" in risk_events.columns:
            risk_events = risk_events[
                risk_events["Title"].astype(str).str.contains(
                    search_text,
                    case=False,
                    na=False,
                )
            ]

        st.divider()
        overview_tab, events_tab, evidence_tab = st.tabs(
            ["Why This Risk Is Changing", "Top Events", "Supporting Evidence"]
        )

        with overview_tab:
            driver_columns = [
                "Acceleration_value",
                "Recency",
                "Source_Accuracy",
                "Impact_Score",
                "Location",
                "Industry_Risk",
                "Frequency_Score",
            ]
            driver_names = {
                "Acceleration_value": "Momentum",
                "Recency": "Recency",
                "Impact_Score": "Impact",
                "Industry_Risk": "Higher-Education Relevance",
                "Location": "Location Relevance",
                "Frequency_Score": "Event Frequency",
                "Source_Accuracy": "Source Reliability",
            }

            available_drivers = [
                column
                for column in driver_columns
                if column in risk_events.columns
            ]

            if available_drivers and not risk_events.empty:
                driver_values = risk_events[available_drivers].apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                driver_summary = (
                    driver_values.mean()
                    .rename("Contribution")
                    .reset_index()
                    .rename(columns={"index": "Technical Driver"})
                )
                driver_summary["Driver"] = driver_summary[
                    "Technical Driver"
                ].map(driver_names)
                driver_summary["Contribution"] = driver_summary[
                    "Contribution"
                ].round(2)
                driver_summary["Level"] = driver_summary[
                    "Contribution"
                ].apply(
                    lambda value: (
                        "High"
                        if value >= 4
                        else "Moderate"
                        if value >= 2.5
                        else "Low"
                    )
                )

                st.dataframe(
                    driver_summary[
                        ["Driver", "Level", "Contribution"]
                    ].sort_values("Contribution", ascending=False),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No driver data is available for this risk.")

        with events_tab:
            if risk_events.empty:
                st.caption("No qualifying events were found for this risk.")
            else:
                risk_events["Published_utc"] = pd.to_datetime(
                    risk_events.get("Published_utc"),
                    errors="coerce",
                    utc=True,
                )
                risk_events["Event_Label_Group"] = risk_events.apply(
                    lambda row: clean_event_label(
                        row.get("Event_Label"),
                        row.get("Title"),
                    ),
                    axis=1,
                )

                grouped_events = (
                    risk_events
                    .groupby("Event_Label_Group", dropna=False)
                    .agg(
                        Raw_Event_Label=("Event_Label", "first"),
                        Event_Severity=("Event_Severity", "max"),
                        Article_Count=("Title", "nunique"),
                        Latest_Date=("Published_utc", "max"),
                        Sample_Title=("Title", "first"),
                        Sample_Content=("Content", "first"),
                        Sample_Link=("Link", "first"),
                    )
                    .reset_index()
                    .sort_values("Event_Severity", ascending=False)
                    .head(maximum_items)
                )

                for _, event_row in grouped_events.iterrows():
                    label = event_row["Event_Label_Group"]
                    severity = safe_number(event_row["Event_Severity"])

                    with st.expander(
                        f"{label} — {severity_bucket(severity)} "
                        f"({severity:.2f})",
                        expanded=False,
                    ):
                        st.caption(
                            f"{int(event_row['Article_Count'])} related "
                            f"article(s) · Latest evidence: "
                            f"{format_date(event_row['Latest_Date'], short=True)}"
                        )

                        sample_title = clean_text(
                            event_row.get("Sample_Title"),
                            "",
                        )
                        if sample_title:
                            st.markdown(
                                f"**Representative headline:** {sample_title}"
                            )

                        sample_content = clean_text(
                            event_row.get("Sample_Content"),
                            "",
                        )
                        if sample_content:
                            preview = sample_content[:700]
                            st.write(
                                preview
                                + ("..." if len(sample_content) > 700 else "")
                            )

                        sample_link = clean_text(
                            event_row.get("Sample_Link"),
                            "",
                        )
                        if sample_link.startswith("http"):
                            st.markdown(
                                f"[Read representative article]({sample_link})"
                            )

        with evidence_tab:
            articles_for_risk = article_data[
                article_data["Dashboard_Risk"].eq(selected_risk)
                & (article_data["Published_utc"] >= cutoff)
            ].copy()

            if selected_subcategory != "All Subcategories":
                articles_for_risk = articles_for_risk[
                    articles_for_risk["Predicted_Risks_new"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .eq(selected_subcategory)
                ]

            if "University Label" in articles_for_risk.columns:
                articles_for_risk["University Label"] = pd.to_numeric(
                    articles_for_risk["University Label"],
                    errors="coerce",
                ).fillna(0).astype(int)
                articles_for_risk = articles_for_risk[
                    articles_for_risk["University Label"].eq(1)
                ]

            if search_text.strip():
                articles_for_risk = articles_for_risk[
                    articles_for_risk["Title"].astype(str).str.contains(
                        search_text,
                        case=False,
                        na=False,
                    )
                ]

            articles_for_risk = (
                articles_for_risk
                .dropna(subset=["Title"])
                .drop_duplicates(subset=["Title", "Link"])
                .sort_values("Published_utc", ascending=False)
                .head(maximum_items)
            )

            if articles_for_risk.empty:
                st.caption("No recent article evidence was found.")
            else:
                for _, article_row in articles_for_risk.iterrows():
                    with st.container(border=True):
                        title = clean_text(
                            article_row.get("Title"),
                            "Untitled article",
                        )
                        link = clean_text(article_row.get("Link"), "")

                        if link.startswith("http"):
                            st.markdown(f"**[{title}]({link})**")
                        else:
                            st.markdown(f"**{title}**")

                        source = clean_text(
                            article_row.get(
                                "Source",
                                article_row.get(
                                    "source",
                                    article_row.get("canonical_source", ""),
                                ),
                            ),
                            "",
                        )
                        published = format_date(
                            article_row.get("Published_utc"),
                            short=True,
                        )
                        metadata = [
                            item
                            for item in [source, published]
                            if item and item != "Not available"
                        ]
                        if metadata:
                            st.caption(" · ".join(metadata))

                        content = clean_text(article_row.get("Content"), "")
                        if content:
                            preview = content[:500]
                            st.write(
                                preview + ("..." if len(content) > 500 else "")
                            )

    # ------------------------------------------------------------------
    # Page layout
    # ------------------------------------------------------------------
    st.title("External Risk Intelligence")
    st.caption(
        "External developments that may affect Tulane's strategy, "
        "operations, compliance, reputation, or financial position."
    )

    overview_tab, developments_tab, explorer_tab = st.tabs(
        [
            "Executive Overview",
            "Developments Requiring Attention",
            "Risk Explorer",
        ]
    )

    with overview_tab:
        render_executive_overview(
            snapshot_data=snapshot,
            development_data=emerging_risk_items,
        )

    with developments_tab:
        render_developments(
            development_data=emerging_risk_items,
        )

    with explorer_tab:
        render_risk_explorer(
            snapshot_data=snapshot,
            event_data=events,
            article_data=articles,
        )    
    
    
        

if selection == "Risk Analysis Dashboard":
    api_key = os.getenv('GEMINI_API_FREE')
    OWNER = 'ERSRisk'
    REPO = 'Tulane-Sentiment-Analysis'
    TAG = 'BERTopic_results'
    ASSET = 'BERTopic_Streamlit.csv.gz'
    client = genai.Client(api_key = api_key)

    @st.cache_data(show_spinner=True, ttl=1800)
    def get_csv_from_release(owner, repo, tag, asset, usecols=None) -> pd.DataFrame:
        token = _github_token()
        if not token:
            raise RuntimeError("GITHUB_TOKEN missing (not injected or empty).")

        headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
        }
        rel = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}",
        headers=headers, timeout=60
        )
        if rel.status_code != 200:
        # show the real reason (401, 404, permissions)
            raise RuntimeError(f"Release lookup {rel.status_code}: {rel.text[:300]}")

        rel_json = rel.json()
        asset_obj = next((a for a in rel_json.get('assets', []) if a.get('name') == asset), None)
        if not asset_obj:
            raise RuntimeError(f"Asset '{asset}' not found in release '{tag}'.")

        url = asset_obj['browser_download_url']
        r = requests.get(url, headers={"Authorization": f"token {token}", "Accept": "application/octet-stream"}, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Asset download {r.status_code}: {r.text[:300]}")
        return pd.read_csv(io.BytesIO(r.content), compression="gzip", low_memory=False, dtype=str, usecols=usecols)
    
    df = load_csv_gz_from_gcs('latest/BERTopic_Streamlit.csv.gz', 'pipeline/resources/BERTopic_Streamlit.csv.gz')
    #df = get_csv_from_release(OWNER, REPO, TAG, ASSET)
    st.set_page_config(layout="wide")
    
    with open('Model_training/topics_BERT.json', 'r', encoding = 'utf-8') as f:
        topic_names = json.load(f)
    
    topic_dict = {item['topic']: item['name'] for item in topic_names['topics']}
    df['Topic'] = pd.to_numeric(df['Topic'], errors = 'coerce')
    df['Topic'] = df['Topic'].round().astype('Int64')
    trash_topics = [95,94,76,75,52,44,17,10,7,0,559,527,515,503,481,474,469,462,
                461,452,450,445,438,434,395,389,354,349,345,323,315,301,299,
                258,257,254,249,236,234,228,224,208,198,191,188,186,178,177,
                174,172,167,164,156,154,140,136,135,130,125,110,101,90,84,73,
                60,59,56,54,50,24,22,18,568,565,550,526,518,505,484,477,458,
                456,387,245,239,226,196,155,144,123,117,109,105,85,61,33,28,
                25,16,14,143, 6]
    df = df[~(df['Topic'].isin(trash_topics))]
    df['Risk_Score'] = pd.to_numeric(df['Risk_Score'].astype(str).str.strip(), errors = 'coerce')
    df['Topic_names'] = df['Topic'].map(topic_dict)
    df['Published_utc'] = pd.to_datetime(df['Published_utc'], errors='coerce', utc=True)
    df = df[['Published_utc', 'Title', 'Predicted_Risks_new', 'Topic', 'Topic_names', 'Risk_Score', 'Location']]
    df = df.drop_duplicates(subset = ['Title'], keep = 'first')
    
    metric_df = df.copy()
    start_date = st.sidebar.date_input('Start date', date.today() - timedelta(days=30))
    end_date = st.sidebar.date_input('End date', date.today())
    filtered_df = df[(df['Published_utc'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['Published_utc'] <= pd.to_datetime(end_date).tz_localize('UTC'))]
    
    today = pd.to_datetime(date.today()).tz_localize('UTC')
    recent_start = today - pd.Timedelta(days=15)
    baseline_start = today - pd.Timedelta(days=30)
    baseline_end = recent_start
    
    view = st.sidebar.radio('Select View', options = ['Risks'] + ['Topics'], index=0, key='detailed_topic_select')
    
    st.title('Risk and Topic Trends Dashboard')
    if view == 'Topics':
        for topic in metric_df['Topic_names'].unique():
            recent_mean = (metric_df[(metric_df['Published_utc'] >= recent_start) & (metric_df['Published_utc'] <= today) & (metric_df['Topic_names'] == topic)].shape[0] / 15)
            baseline_mean = (metric_df[(metric_df['Published_utc'] >= baseline_start) & (metric_df['Published_utc'] < baseline_end) & (metric_df['Topic_names'] == topic)].shape[0] / 15)
            if baseline_mean == 0:
                if recent_mean == 0:
                    percent_change = 0
                continue
            percent_change = ((recent_mean - baseline_mean) / baseline_mean * 100 if baseline_mean != 0 else float('inf'))
            if percent_change >= 50:
                st.warning(f'Topic "{topic}" is experiencing a significant increase in article volume: {percent_change:.2f}% increase compared to baseline.', icon="⚠️")
        with st.expander('View Full Topics Trend', expanded = False):
            topic_rows = []
            for topic in metric_df['Topic_names'].unique():
                recent_count = metric_df[(metric_df['Published_utc'] >= recent_start) & (metric_df['Published_utc'] <= today) & (metric_df['Topic_names'] == topic)].shape[0]
                baseline_count = metric_df[(metric_df['Published_utc'] >= baseline_start) & (metric_df['Published_utc'] < baseline_end) & (metric_df['Topic_names'] == topic)].shape[0]
                recent_mean = recent_count / 30
                baseline_mean = baseline_count / 30
                
                if baseline_mean == 0:
                    if recent_mean == 0:
                        continue
                    status = 'Emerging'
                else:
                    percentage_change = ((recent_mean - baseline_mean) / baseline_mean * 100) 
                
                    if percentage_change >= 50:
                        status = 'Rising'
                    elif percentage_change <= -30:
                        status = 'Falling'
                    else:
                        status = 'Stable'
                topic_rows.append({
                    'Topic': topic,
                    'Recent Count (Last 30 days)': recent_count,
                    'Baseline Count (30-90 days ago)': baseline_count,
                    'Percentage Change (%)': f'{percentage_change:.2f}%' if baseline_mean != 0 else 'N/A',
                    'Status': status
                })
            topic_trend_df = pd.DataFrame(topic_rows)
            st.dataframe(topic_trend_df, use_container_width=True, hide_index = True)
    if view == 'Risks':
        for risk in metric_df['Predicted_Risks_new'].unique():
            if risk == 'No Risk':
                continue
            recent_mean = (metric_df[(metric_df['Published_utc'] >= recent_start) & (metric_df['Published_utc'] <= today) & (metric_df['Predicted_Risks_new'] == risk)].shape[0] / 30)
            baseline_mean = (metric_df[(metric_df['Published_utc'] >= baseline_start) & (metric_df['Published_utc'] < baseline_end) & (metric_df['Predicted_Risks_new'] == risk)].shape[0] / 30)
            if baseline_mean == 0:
                if recent_mean == 0:
                    percent_change = 0
                continue
            percent_change = ((recent_mean - baseline_mean) / baseline_mean * 100 if baseline_mean != 0 else float('inf'))
            if percent_change >= 50:
                st.warning(f'Risk "{risk}" is experiencing a significant increase in article volume: {percent_change:.2f}% increase compared to baseline.', icon="⚠️")
        with st.expander('View Full Topics Trend', expanded = False):
            risk_rows = []
            for risk in metric_df['Predicted_Risks_new'].unique():
                if risk == 'No Risk':
                    continue
                recent_count = metric_df[(metric_df['Published_utc'] >= recent_start) & (metric_df['Published_utc'] <= today) & (metric_df['Predicted_Risks_new'] == risk)].shape[0]
                baseline_count = metric_df[(metric_df['Published_utc'] >= baseline_start) & (metric_df['Published_utc'] < baseline_end) & (metric_df['Predicted_Risks_new'] == risk)].shape[0]
                recent_mean = recent_count / 30
                baseline_mean = baseline_count / 30
                
                if baseline_mean == 0:
                    if recent_mean == 0:
                        continue
                    status = 'Emerging'
                else:
                    percentage_change = ((recent_mean - baseline_mean) / baseline_mean * 100) 
                
                    if percentage_change >= 50:
                        status = 'Rising'
                    elif percentage_change <= -30:
                        status = 'Falling'
                    else:
                        status = 'Stable'
                risk_rows.append({
                    'Risk': risk,
                    'Recent Count (Last 30 days)': recent_count,
                    'Baseline Count (30-90 days ago)': baseline_count,
                    'Percentage Change (%)': f'{percentage_change:.2f}%' if baseline_mean != 0 else 'N/A',
                    'Status': status
                })
            risk_trend_df = pd.DataFrame(risk_rows)
            st.dataframe(risk_trend_df.sort_values(by='Percentage Change (%)', ascending=False), use_container_width=True, hide_index = True)
                
    
    
    col1, col2 = st.columns(2)
    with col1:
        aggregation = st.radio('Select Metric Aggregation', ['Count', 'Cumulative', 'Weighted Severity'], index=0, key='metric_aggregation')
    with col2:
        frequency = st.radio('Select Time Frequency', ['Daily', 'Weekly'], index=1, key='time_frequency')
        if frequency == 'Daily':
            freq = 'D'
        elif frequency == 'Weekly':
            freq = 'W'
    
    #give me a line chart that shows the count of articles per day for each topic over time
    
    
    # Drop rows where date or topic is missing
    filtered_df = filtered_df.dropna(subset=['Published_utc', 'Topic'])
    
    if aggregation == 'Weighted Severity':
        weighted_df = filtered_df.groupby(['Topic_names', pd.Grouper(key='Published_utc', freq = freq)])['Risk_Score'].sum().reset_index(name = 'Weighted_Severity')
        top_topics = (weighted_df.groupby('Topic_names')['Weighted_Severity']
                    .sum()
                    .nlargest(5)
                    .index)
        data = weighted_df[weighted_df['Topic_names'].isin(top_topics)]
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Topic Trends Over Time')
            chart = (
                alt.Chart(data)
                .mark_line(point = True)
                .encode(
                    x = alt.X('Published_utc:T', title='Date'),
                    y = alt.Y('Weighted_Severity:Q', title='Weighted Severity'),
                    color = alt.Color('Topic_names:N', title='Topic', legend=alt.Legend(orient='bottom', direction = 'vertical', labelLimit = 0, labelFontSize=12, symbolSize=100) ),
                )
                .properties(width=600, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
        risk_trends = filtered_df.groupby(['Predicted_Risks_new', pd.Grouper(key='Published_utc', freq=freq)])['Risk_Score'].sum().reset_index(name = 'Weighted_Severity')
        recent_risk_data = risk_trends.copy()
        recent_risk_data = recent_risk_data[~(recent_risk_data['Predicted_Risks_new'] == 'No Risk')]
        top_risks = (recent_risk_data.groupby('Predicted_Risks_new')['Weighted_Severity']
                    .sum()
                    .nlargest(5)
                    .index)
        recent_risk_data = recent_risk_data[recent_risk_data['Predicted_Risks_new'].isin(top_risks)]
        with col2:
            st.subheader('Risk Trends Over Time')
            chart = (
                alt.Chart(recent_risk_data)
                .mark_line(point = True)
                .encode(
                    x = alt.X('Published_utc:T', title='Date'),
                    y = alt.Y('Weighted_Severity:Q', title='Weighted Severity'),
                    color = alt.Color('Predicted_Risks_new:N', title='Risk', legend=alt.Legend(orient='bottom', direction='vertical', labelLimit = 0, labelFontSize=12, symbolSize=100) ),
                )
                .properties(width=600, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
    
    if aggregation == 'Count':
        daily_counts = filtered_df.groupby(['Topic_names', pd.Grouper(key='Published_utc', freq=freq)]).size().reset_index(name='Article_Count')
        recent_data = daily_counts.copy()
        #keep only top 5 topics by total article count in recent data
        top_topics = (recent_data.groupby('Topic_names')['Article_Count']
                    .sum()
                    .nlargest(5)
                    .index)
        data = recent_data[recent_data['Topic_names'].isin(top_topics)]
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Topic Trends Over Time')
            chart = (
                alt.Chart(data)
                .mark_line(point = True)
                .encode(
                    x = alt.X('Published_utc:T', title='Date'),
                    y = alt.Y('Article_Count:Q', title='Number of Articles'),
                    color = alt.Color('Topic_names:N', title='Topic', legend=alt.Legend(orient='bottom', direction = 'vertical', labelLimit = 0, labelFontSize=12, symbolSize=100) ),
                )
                .properties(width=600, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
    
        risk_trends = filtered_df.groupby(['Predicted_Risks_new', pd.Grouper(key='Published_utc', freq=freq)]).size().reset_index(name='Article_Count')
        recent_risk_data = risk_trends.copy()
        recent_risk_data = recent_risk_data[~(recent_risk_data['Predicted_Risks_new'] == 'No Risk')]
        top_risks = (recent_risk_data.groupby('Predicted_Risks_new')['Article_Count']
                    .sum()
                    .nlargest(5)
                    .index)
        recent_risk_data = recent_risk_data[recent_risk_data['Predicted_Risks_new'].isin(top_risks)]
        with col2:
            st.subheader('Risk Trends Over Time')
            chart = (
                alt.Chart(recent_risk_data)
                .mark_line(point = True)
                .encode(
                    x = alt.X('Published_utc:T', title='Date'),
                    y = alt.Y('Article_Count:Q', title='Number of Articles'),
                    color = alt.Color('Predicted_Risks_new:N', title='Risk', legend=alt.Legend(orient='bottom', direction='vertical', labelLimit = 0, labelFontSize=12, symbolSize=100) ),
                )
                .properties(width=600, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
    
    if aggregation == 'Cumulative':
        cumulative_counts = filtered_df.groupby(['Topic_names', pd.Grouper(key='Published_utc', freq=freq)]).size().groupby(level=0).cumsum().reset_index(name='Cumulative_Article_Count')
        recent_data = cumulative_counts.copy()
        #keep only top 5 topics by total article count in recent data
        top_topics = (recent_data.groupby('Topic_names')['Cumulative_Article_Count']
                    .max()
                    .nlargest(5)
                    .index)
        data = recent_data[recent_data['Topic_names'].isin(top_topics)]
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Cumulative Topic Trends Over Time')
            chart = (
                alt.Chart(data)
                .mark_line(point = True)
                .encode(
                    x = alt.X('Published_utc:T', title='Date'),
                    y = alt.Y('Cumulative_Article_Count:Q', title='Cumulative Number of Articles'),
                    color = alt.Color('Topic_names:N', title='Topic', legend=alt.Legend(orient='bottom', direction = 'vertical', labelLimit = 0, labelFontSize=12, symbolSize=100) ),
                )
                .properties(width=600, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
    
        risk_trends = filtered_df.groupby(['Predicted_Risks_new', pd.Grouper(key='Published_utc', freq=freq)]).size().groupby(level=0).cumsum().reset_index(name='Cumulative_Article_Count')
        recent_risk_data = risk_trends.copy()
        recent_risk_data = recent_risk_data[~(recent_risk_data['Predicted_Risks_new'] == 'No Risk')]
        top_risks = (recent_risk_data.groupby('Predicted_Risks_new')['Cumulative_Article_Count']
                    .max()
                    .nlargest(5)
                    .index)
        recent_risk_data = recent_risk_data[recent_risk_data['Predicted_Risks_new'].isin(top_risks)]
        with col2:
            st.subheader('Cumulative Risk Trends Over Time')
            chart = (
                alt.Chart(recent_risk_data)
                .mark_line(point = True)
                .encode(
                    x = alt.X('Published_utc:T', title='Date'),
                    y = alt.Y('Cumulative_Article_Count:Q', title='Cumulative Number of Articles'),
                    color = alt.Color('Predicted_Risks_new:N', title='Risk', legend=alt.Legend(orient='bottom', direction='vertical', labelLimit = 0, labelFontSize=12, symbolSize=100) ),
                )
                .properties(width=600, height=400)
            )
            st.altair_chart(chart, use_container_width=True)
    
    def sparkline(data, ycol = 'Value'):
            return(alt.Chart(data).mark_area(opacity = 0.25).encode(x= alt.X('Date:T'), y= alt.Y(f'{ycol}:Q')).properties(height=40, width=150))
    st.markdown("""
        <style>
        .kpi-card > div { padding: 1.2rem 1rem; border:1px solid #313131; border-radius:12px; }
        .kpi-title { margin:0; font-size:0.9rem; color:#9aa0a6; }
        .kpi-value { font-size:1.8rem; font-weight:700; line-height:1; margin: 0.25rem 0 0.5rem; }
        .kpi-delta { font-size:0.9rem; opacity:0.85; margin-bottom:0.25rem; }
        .kpi-spacer { height: 4px; }
        </style>
        """, unsafe_allow_html=True)
    
    def kpi_card(title, value, delta_text=None, data=None, axis='Value'):
        with st.container(border=False):
            # apply visual shell to the *inner* container only
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            with st.container(border=True):
                # header
                st.markdown(f'<div class="kpi-title">{title}</div>', unsafe_allow_html=True)
                # value + optional delta (no arrow)
                cols = st.columns([3,1])
                with cols[0]:
                    st.markdown(f'<div class="kpi-value">{value}</div>', unsafe_allow_html=True)
                    if delta_text:
                        st.markdown(f'<div class="kpi-delta">{delta_text}</div>', unsafe_allow_html=True)
                with cols[1]:
                    st.empty()
                # sparkline (stays inside because it's inside this container)
                if data is not None and not data.empty:
                    st.altair_chart(sparkline(data, ycol=axis), use_container_width=True)
                else:
                    st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    if view == 'Topics':
        topic = st.selectbox('Select Topic', options = sorted(filtered_df['Topic_names'].unique()), key='selected_topic')
        st.subheader(f'Topic: {topic}')
    
    
        recent_mean = (metric_df[(metric_df['Published_utc'] >= recent_start) & (metric_df['Published_utc'] <= today) & (metric_df['Topic_names'] == topic)].shape[0] / 30)
        baseline_mean = (metric_df[(df['Published_utc'] >= baseline_start) & (metric_df['Published_utc'] < baseline_end) & (metric_df['Topic_names'] == topic)].shape[0] / 30)
        percent_change = ((recent_mean - baseline_mean) / baseline_mean * 100 if baseline_mean != 0 else float('inf'))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            kpi_card("Articles in Last 30 Days", f"{recent_mean * 30:.0f}")
        with col2:
            if percent_change >= 0.2 and percent_change != float('inf'):
                kpi_card("30-Day Change", f"{percent_change:.2f}%", delta_text = "Increasing")
            elif percent_change <= -0.2 and percent_change != float('inf'):
                kpi_card("30-Day Change", f"{percent_change:.2f}%", delta_text = "Decreasing")
            elif percent_change == float('inf'):
                kpi_card("30-Day Change", "N/A")
        with col3:
            average_risk_score = filtered_df[filtered_df['Topic_names'] == topic]['Risk_Score'].mean()
            kpi_card("Average Risk Score", f"{average_risk_score:.2f}")
        with col4:
            local_mask = filtered_df[(filtered_df['Topic_names'] == topic) & (filtered_df['Location'].fillna(0).astype(int) >= 3)].shape[0]
            kpi_card("Local Mentions", f"{local_mask}")
    
    
        if aggregation == 'Count':
            daily_counts = filtered_df.groupby(['Topic_names', pd.Grouper(key='Published_utc', freq=freq)]).size().reset_index(name='Article_Count')
            topic_data = daily_counts[daily_counts['Topic_names'] == topic]
            value_col = '__count__'
    
        if aggregation == 'Cumulative':
            cumulative_counts = filtered_df.groupby(['Topic_names', pd.Grouper(key='Published_utc', freq=freq)]).size().groupby(level=0).cumsum().reset_index(name='Cumulative_Article_Count')
            topic_data = cumulative_counts[cumulative_counts['Topic_names'] == topic]
            value_col = 'Cumulative'
    
        if aggregation == 'Weighted Severity':
            weighted_df = filtered_df.groupby(['Topic_names', pd.Grouper(key='Published_utc', freq = freq)])['Risk_Score'].sum().reset_index(name = 'Weighted_Severity')
            topic_data = weighted_df[weighted_df['Topic_names'] == topic]
            value_col = 'Risk_Score'
    
    
        def prepare_series(df, group_col, value_col, start = None, end = None, freq = freq):
            s = df.copy()
            s['Published_utc'] = pd.to_datetime(s['Published_utc'], errors='coerce', utc=True).dt.normalize()
            if start is not None:
                s = s[s['Published_utc'] >= pd.to_datetime(date.today() - timedelta(days=180)).tz_localize('UTC')]
            if end is not None:
                s = s[s['Published_utc'] <= pd.to_datetime(end).tz_localize('UTC')]
            
            if value_col == '__count__':
                ts = (s.groupby(pd.Grouper(key='Published_utc', freq=freq)).size().rename('Value').to_frame())
            if value_col == 'Risk_Score':
                ts = (s.groupby(pd.Grouper(key='Published_utc', freq=freq))[value_col].sum().rename('Value').to_frame())
            if value_col == 'Cumulative':
                ts = (s.groupby(pd.Grouper(key='Published_utc', freq=freq)).size().rename('Value').to_frame())
                ts['Value'] = ts['Value'].cumsum()
            
            today = pd.to_datetime(date.today()).tz_localize('UTC')
            #if not ts.empty:
                #full_idx = pd.date_range(start=ts.index.min(), end=today, freq=freq, tz='UTC')
                #ts = ts.reindex(full_idx, fill_value=0)
            ts.index.name = 'Date'
            if freq == 'D':
                window_size = 7
            else:
                window_size = 4
            
            ts = ts.reset_index()
            return ts
    
        def forecast_ets(ts_df, horizon = 2, seasonal = None):
            if ts_df.empty or len(ts_df) < 4:
                return ts_df.assign(kind='actual'), pd.DataFrame(), None
            y = ts_df['Value'].astype(float).values
    
            model = ExponentialSmoothing(y, trend='add', seasonal=seasonal, seasonal_periods=None).fit(optimized = True)
            fitted_vals = model.fittedvalues
            fcst_vals = model.forecast(horizon)
    
    
            #Approximate 95% CI from residuals
            resid = model.resid
            sigma = np.nanstd(resid)
            lower = fcst_vals - 1.96 * sigma
            upper = fcst_vals + 1.96 * sigma
    
            last_date = ts_df['Date'].iloc[-1]
            inferred = pd.infer_freq(ts_df['Date'])
            freq = inferred if inferred else 'W'
            future_idx = pd.date_range(start=last_date, periods = horizon, freq = freq)
            fcst_df = pd.DataFrame({
                'Date': future_idx,
                'yhat': fcst_vals,
                'yhat_lower': np.maximum(0, lower),
                'yhat_upper': np.maximum(0, upper),
                'kind': 'forecast'
            }) 
            actual_df = ts_df.copy()
            actual_df['kind'] = 'actual'
            cutoff = last_date
            return actual_df, fcst_df, cutoff
    
        def layered_forecast_chart(actual_df, fcst_df, title =None):
            actual_line = alt.Chart(actual_df).mark_line(strokeWidth=2, point = True).encode(
                x = 'Date:T', y = 'Value:Q',
                color = alt.value('lightblue'),
                tooltip = [alt.Tooltip('Date:T'), alt.Tooltip('Value:Q', format = ',')]
            ).properties(title=title)
    
            
            layers = [actual_line]
    
            if fcst_df.empty:
                return actual_line
            
            band = alt.Chart(fcst_df).mark_area(opacity=0.15).encode(
                x = 'Date:T',
                y = 'yhat_lower:Q',
                y2 = 'yhat_upper:Q',
                color = alt.value('orange')
            )
            fcst_line = alt.Chart(fcst_df).mark_line(strokeDash=[6,4], strokeWidth=2).encode(
                x = 'Date:T',
                y = 'yhat:Q',
                color = alt.value('orange'),
                tooltip = [alt.Tooltip('Date:T'), alt.Tooltip('yhat:Q', format = ',')]
            )
            layers.append(band)
            layers.append(fcst_line)
            return alt.layer(*layers).properties(title=title).resolve_scale(y='shared')
    
        forecast_on = st.checkbox('Enable Forecasting', value = True, key='forecast_toggle')
        ts = prepare_series(filtered_df[filtered_df['Topic_names'] == topic], group_col = 'Topic_names', value_col = value_col, start = start_date, end = end_date, freq = freq)
        if forecast_on:
            actual_df, fcst_df, cutoff = forecast_ets(ts, horizon = 2)
            title = f'Forecast for Topic: {topic}'
        else:
            actual_df = ts.copy()
            actual_df['kind'] = 'actual'
            fcst_df = pd.DataFrame()
    
            title = f'Time Series for Topic: {topic}'
    
        chart = layered_forecast_chart(actual_df, fcst_df, title=title)
        st.altair_chart(chart, use_container_width=True)
    
        if st.button('Generate AI Summary', key = 'AI_insight'):
            prompt = f"Provide a concise summary of the recent trends for the topic '{topic}' based on the articles found here: {filtered_df[filtered_df['Topic_names'] == topic]['Title'].tolist()}. Highlight any significant trends or changes in sentiment and focus your response to higher education risks this topic poses. Also, provide the titles of the articles you reference. Keep your responses less than 200 words long."
            response = client.models.generate_content(
                model = 'gemini-2.5-flash',
                contents = prompt
            )
            st.subheader('AI-Generated Summary')
            st.write(response.text)
        
        def associated_risks(topic):
            risks = filtered_df[filtered_df['Topic_names'] == topic]['Predicted_Risks_new'].value_counts()
            return risks
    
        st.subheader('Associated Risks')
        data = associated_risks(topic).reset_index()
        chart = alt.Chart(data).mark_bar().encode(
            y = alt.Y('Predicted_Risks_new:N', title='Risk', axis=alt.Axis(labelLimit=0)),
            x = alt.X('count:Q', title='Count of Articles'),
            tooltip = [alt.Tooltip('Predicted_Risks_new:N', title='Risk'),
                    alt.Tooltip('count:Q', title='Count')]
        ).properties(width=800, height=max(240, 22*len(data)))
        st.altair_chart(chart, use_container_width=True)
    
    if view == 'Risks':
        risk = st.selectbox('Select Topic', options = sorted(filtered_df['Predicted_Risks_new'].unique()), key='selected_risk')
        st.subheader(f'Risk: {risk}')
    
    
        recent_mean = (metric_df[(metric_df['Published_utc'] >= recent_start) & (metric_df['Published_utc'] <= today) & (metric_df['Predicted_Risks_new'] == risk)].shape[0] / 30)
        baseline_mean = (metric_df[(df['Published_utc'] >= baseline_start) & (metric_df['Published_utc'] < baseline_end) & (metric_df['Predicted_Risks_new'] == risk)].shape[0] / 30)
        percent_change = ((recent_mean - baseline_mean) / baseline_mean * 100 if baseline_mean != 0 else float('inf'))
    
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            kpi_card("Articles in Last 30 Days", f"{recent_mean * 30:.0f}")
        with col2:
            if percent_change >= 0.2 and percent_change != float('inf'):
                kpi_card("30-Day Change", f"{percent_change:.2f}%", delta_text = "Increasing")
            elif percent_change <= -0.2 and percent_change != float('inf'):
                kpi_card("30-Day Change", f"{percent_change:.2f}%", delta_text = "Decreasing")
            elif percent_change == float('inf'):
                kpi_card("30-Day Change", "N/A")
        with col3:
            average_risk_score = filtered_df[filtered_df['Predicted_Risks_new'] == risk]['Risk_Score'].mean()
            kpi_card("Average Risk Score", f"{average_risk_score:.2f}")
        with col4:
            local_mask = filtered_df[(filtered_df['Predicted_Risks_new'] == risk) & (filtered_df['Location'].fillna(0).astype(int) >= 3)].shape[0]
            kpi_card("Local Mentions", f"{local_mask}")
    
        if aggregation == 'Count':
            daily_counts = filtered_df.groupby(['Predicted_Risks_new', pd.Grouper(key='Published_utc', freq=freq)]).size().reset_index(name='Article_Count')
            topic_data = daily_counts[daily_counts['Predicted_Risks_new'] == risk]
            value_col = '__count__'
    
        if aggregation == 'Cumulative':
            cumulative_counts = filtered_df.groupby(['Predicted_Risks_new', pd.Grouper(key='Published_utc', freq=freq)]).size().groupby(level=0).cumsum().reset_index(name='Cumulative_Article_Count')
            topic_data = cumulative_counts[cumulative_counts['Predicted_Risks_new'] == risk]
            value_col = 'Cumulative'
        if aggregation == 'Weighted Severity':
            weighted_df = filtered_df.groupby(['Predicted_Risks_new', pd.Grouper(key='Published_utc', freq = freq)])['Risk_Score'].sum().reset_index(name = 'Weighted_Severity')
            topic_data = weighted_df[weighted_df['Predicted_Risks_new'] == risk]
            value_col = 'Risk_Score'
    
    
        def prepare_series(df, group_col, value_col, start = None, end = None, freq = freq):
            s = df.copy()
            s['Published_utc'] = pd.to_datetime(s['Published_utc'], errors='coerce', utc=True).dt.normalize()
            if start is not None:
                s = s[s['Published_utc'] >= pd.to_datetime(date.today() - timedelta(days=180)).tz_localize('UTC')]
            if end is not None:
                s = s[s['Published_utc'] <= pd.to_datetime(end).tz_localize('UTC')]
            
            if value_col == '__count__':
                ts = (s.groupby(pd.Grouper(key='Published_utc', freq=freq)).size().rename('Value').to_frame())
            if value_col == 'Cumulative':
                ts = (s.groupby(pd.Grouper(key='Published_utc', freq=freq)).size().rename('Value').to_frame())
                ts['Value'] = ts['Value'].cumsum()
            if value_col == 'Risk_Score':
                ts = (s.groupby(pd.Grouper(key='Published_utc', freq=freq))[value_col].sum().rename('Value').to_frame())
            
            today = pd.to_datetime(date.today()).tz_localize('UTC')
            #if not ts.empty:
                #full_idx = pd.date_range(start=ts.index.min(), end=today, freq=freq, tz='UTC')
                #ts = ts.reindex(full_idx, fill_value=0)
            ts.index.name = 'Date'
            if freq == 'D':
                window_size = 7
            else:
                window_size = 4
            
            ts = ts.reset_index()
            return ts
    
        def forecast_ets(ts_df, horizon = 2, seasonal = None):
            if ts_df.empty or len(ts_df) < 4:
                return ts_df.assign(kind='actual'), pd.DataFrame(), None
            y = ts_df['Value'].astype(float).values
    
            model = ExponentialSmoothing(y, trend='add', seasonal=seasonal, seasonal_periods=None).fit(optimized = True)
            fitted_vals = model.fittedvalues
            fcst_vals = model.forecast(horizon)
    
    
            #Approximate 95% CI from residuals
            resid = model.resid
            sigma = np.nanstd(resid)
            lower = fcst_vals - 1.96 * sigma
            upper = fcst_vals + 1.96 * sigma
    
            last_date = ts_df['Date'].iloc[-1]
            inferred = pd.infer_freq(ts_df['Date'])
            freq = inferred if inferred else 'W'
            future_idx = pd.date_range(start=last_date, periods = horizon, freq = freq)
            fcst_df = pd.DataFrame({
                'Date': future_idx,
                'yhat': fcst_vals,
                'yhat_lower': np.maximum(0, lower),
                'yhat_upper': np.maximum(0, upper),
                'kind': 'forecast'
            }) 
            actual_df = ts_df.copy()
            actual_df['kind'] = 'actual'
            cutoff = last_date
            return actual_df, fcst_df, cutoff
    
        def layered_forecast_chart(actual_df, fcst_df, title =None):
            actual_line = alt.Chart(actual_df).mark_line(strokeWidth=2, point = True).encode(
                x = 'Date:T', y = 'Value:Q',
                color = alt.value('lightblue'),
                tooltip = [alt.Tooltip('Date:T'), alt.Tooltip('Value:Q', format = ',')]
            ).properties(title=title)
    
            
            layers = [actual_line]
    
            if fcst_df.empty:
                return actual_line
            
            band = alt.Chart(fcst_df).mark_area(opacity=0.15).encode(
                x = 'Date:T',
                y = 'yhat_lower:Q',
                y2 = 'yhat_upper:Q',
                color = alt.value('orange')
            )
            fcst_line = alt.Chart(fcst_df).mark_line(strokeDash=[6,4], strokeWidth=2).encode(
                x = 'Date:T',
                y = 'yhat:Q',
                color = alt.value('orange'),
                tooltip = [alt.Tooltip('Date:T'), alt.Tooltip('yhat:Q', format = ',')]
            )
            layers.append(band)
            layers.append(fcst_line)
            return alt.layer(*layers).properties(title=title).resolve_scale(y='shared')
    
        forecast_on = st.checkbox('Enable Forecasting', value = True, key='forecast_toggle')
        ts = prepare_series(filtered_df[filtered_df['Predicted_Risks_new'] == risk], group_col = 'Predicted_Risks_new', value_col = value_col, start = start_date, end = end_date, freq = freq)
        if forecast_on:
            actual_df, fcst_df, cutoff = forecast_ets(ts, horizon = 2)
            title = f'Forecast for Risk: {risk}'
        else:
            actual_df = ts.copy()
            actual_df['kind'] = 'actual'
            fcst_df = pd.DataFrame()
    
            title = f'Time Series for Topic: {topic}'
    
        chart = layered_forecast_chart(actual_df, fcst_df, title=title)
        st.altair_chart(chart, use_container_width=True)
    
        if st.button('Generate AI Summary', key = 'AI_insight_risk'):
            prompt = f"Provide a concise summary of the recent trends for the risk '{risk}' based on the articles found here: {filtered_df[filtered_df['Predicted_Risks_new'] == risk]['Title'].tolist()}. Highlight any significant trends or changes in sentiment and focus your response to higher education risks this topic poses. Also, provide the titles of the articles you reference. Keep your responses less than 200 words long."
            response = client.models.generate_content(
                model = 'gemini-2.5-flash',
                contents = prompt
            )
            st.subheader('AI-Generated Summary')
            st.write(response.text)
    
        def associated_topics(risk):
            risks = filtered_df[filtered_df['Predicted_Risks_new'] == risk]['Topic_names'].value_counts()
            return risks
    
        st.subheader('Associated Risks')
        data = associated_topics(risk).reset_index()
        chart = alt.Chart(data).mark_bar().encode(
            y = alt.Y('Topic_names:N', title='Topic', axis=alt.Axis(labelLimit=0)),
            x = alt.X('count:Q', title='Count of Articles'),
            tooltip = [alt.Tooltip('Topic_names:N', title='Topic'),
                    alt.Tooltip('count:Q', title='Count')]
        ).properties(width=800, height=max(240, 22*len(data)))
        st.altair_chart(chart, use_container_width=True)



if selection == "Unmatched Topic Analysis":
    from typing import Iterable, Any
    def push_file_to_github(local_path:str, repo:str, dest_path:str, branch:str = "main", token:str|None = None):
        token = os.getenv('GITHUB_TOKEN')
        try:
            with open(local_path, "rb") as f:
                content_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            default_bytes = b"[]\n" if local_path.lower().endswith(".json") else b""
            with open(local_path, 'wb') as f:
                f.write(default_bytes)
            content_bytes = default_bytes
            content_b64 = base64.b64encode(content_bytes).decode("utf-8")


        api_base = f"https://api.github.com/repos/{repo}/contents/{dest_path}"
        headers = {"Authorization": f"token {token}", "Accept":"application/vnd.github+json"}

        sha = None
        r_get = requests.get(api_base, headers = headers, params = {"ref":branch})
        if r_get.status_code == 200:
            sha = r_get.json()['sha']
        payload = {
            "message": f"Update {dest_path} via Streamlit at {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "content": content_b64,
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        r_put = requests.put(api_base, headers = headers, data = json.dumps(payload))
        if r_put.status_code not in (200, 201):
            raise RuntimeError(f"GitHub push failed: {r_put.status_code} {r_put.text}")
        return r_put.json()

    def fetch_release(owner, repo, tag:str, asset_name:str, token:str):
        headers = {'Authorization': f'token {token}',
                  'Accept': 'application/vnd.github+json'}
        r = requests.get(f'https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}', headers=headers, timeout=60)
        if r.status_code == 404:
            r = requests.post(f'https://api.github.com/repos/{owner}/{repo}/releases', headers = headers, timeout = 60, json={
            "tag_name": tag, "name": tag, "draft": False, "prerelease": False
        })
        r.raise_for_status()
        rel = r.json()
        upload_url = rel['upload_url'].split('{', 1)[0]
        assets = requests.get(f"https://api.github.com/repos/{owner}/{repo}/releases/{rel['id']}/assets", headers=headers, timeout=60).json()
        a = next((x for x in assets if x.get("name") == asset_name), None)
        if not a:
            return []

        url = a.get('browser_download_url')
        b = requests.get(url, headers = headers, timeout = 120)
        b.raise_for_status()
        content = b.content
        return json.loads(content.decode('utf-8'))

    def upload_asset_to_release(owner, repo, tag:str, asset_path:str, token:str):
        headers = {'Authorization': f'token {token}',
                  'Accept': 'application/vnd.github+json'}
        r = requests.get(f'https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}', headers=headers, timeout=60)
        r.raise_for_status()
        rel = r.json()
        upload_url = rel["upload_url"].split("{", 1)[0]
        assets = requests.get(f"https://api.github.com/repos/{owner}/{repo}/releases/{rel['id']}/assets", headers=headers, timeout=60).json()
        name = os.path.basename(asset_path)
        for a in assets:
            if a.get("name") == name:
                requests.delete(f"https://api.github.com/repos/{owner}/{repo}/releases/assets/{a['id']}", headers=headers, timeout=60)
        with open(asset_path, "rb") as f:
            up = requests.post(
                f"{upload_url}?name={name}",
                headers={"Authorization": f"token {token}", "Content-Type": "application/octet-stream"},
                data=f.read(), timeout=300
            )
        up.raise_for_status()
        return up.json()
    def upsert_single_big_json(owner, repo, tag: str, asset_name: str,
                           new_items: list, dedupe_key: str, token: str, mode = 'merge'):
        if mode == 'replace':
            current = []
        else:
            current = fetch_release(owner, repo, tag, asset_name, token)
            if not isinstance(current, list):
                current = []

        # 2) merge by key (new replaces old on same key)
        by_key = {}
        for it in (current if mode == "merge" else []):
            k = it.get(dedupe_key)
            if k is not None:
                by_key[k] = it
        for it in new_items:
            k = it.get(dedupe_key)
            if k is not None:
                by_key[k] = it
        merged = list(by_key.values())

        # 3) write to a temp gz and upload (same asset name → old is deleted then replaced)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, asset_name)  # e.g., "unmatched_topics.json.gz"
            raw = json.dumps(merged, ensure_ascii=False).encode("utf-8")
            if asset_name.endswith(".gz"):
                with gzip.open(path, "wb") as f:
                    f.write(raw)
            else:
                with open(path, "wb") as f:
                    f.write(raw)
            return upload_asset_to_release(owner, repo, tag, path, token)


    def next_topic_id(existing_ids: Iterable[Any], start: int = 0) -> int:
        ints = [x for x in existing_ids]
        return (max(ints) + 1) if ints else start

    if 'unmatched' not in st.session_state:
        st.session_state.unmatched = fetch_release(
            "ERSRisk", "tulane-sentiment-app-clean",
            "unmatched-topics", "unmatched_topics.json",
            os.getenv('GITHUB_TOKEN')
            ) or []

    if 'topicsbert' not in st.session_state:
        if os.path.exists('Model_training/topics_BERT.json'):
            with open('Model_training/topics_BERT.json', 'r') as f:
                st.session_state.topicsbert = json.load(f)
        else:
            st.session_state.topicsbert = {'topics': []}

    if 'topicandsubtopic' not in st.session_state:
        if os.path.exists('Model_training/topics_BERT_auto.json'):
            with open('Model_training/topics_BERT_auto.json', 'r') as f:
                st.session_state.topicandsubtopic = json.load(f)
        else:
            st.session_state.topicandsubtopic = []

    if 'discarded' not in st.session_state:
        st.session_state.discarded = fetch_release(
            "ERSRisk", "tulane-sentiment-app-clean",
            "discarded-topics", "discarded_topics.json",
            os.getenv('GITHUB_TOKEN')
            ) or []

    st.title('Unmatched Topics Analysis')
    PAGE_SIZE = st.sidebar.selectbox('Items per Page', [10, 20, 30, 50], index =1)
    total = len(st.session_state.unmatched)
    max_page = max(1, (total + PAGE_SIZE - 1)//PAGE_SIZE)

    if 'page_num' not in st.session_state:
        st.session_state.page_num = 1
    st.session_state.page_num = st.sidebar.number_input(
        'Page', min_value = 1, max_value = max_page, value = st.session_state.page_num, step =1
    )

    start = (st.session_state.page_num - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    st.caption(f"Showing {start + 1} to {min(end, total)} of {total} topics")
    page_df = st.session_state.unmatched[start:end]

    for topic in page_df:
        skip_key = f"skip_{topic['topic']}"
        if st.session_state.get(skip_key):
            continue

        st.subheader(f"Topic {topic['topic']}: {topic['name']}")
        st.markdown(f"**Keywords:** {(topic['keywords'])}")
        with st.expander("**Sample Articles:**"):
            docs = topic['documents']
            random.shuffle(docs)
            for doc in docs:
                words = doc.split()
                st.markdown("**Sample Titles:**")
                st.markdown(f"{' '.join(words[:40]) + '...' if len(words)>40 else ''}")
        radio_key = str(topic['topic'])
        reset_flag = f"reset_{radio_key}"


        if st.session_state.get(reset_flag):
            st.session_state[radio_key] = ''
            st.session_state[reset_flag] = False
        decision = st.radio("What would you like to do with this topic?",['','Keep as new topic', 'Merge with existing topic', 'Discard'],
            key=radio_key, index = 0)
        if decision == 'Keep as new topic':
            st.session_state['confirm_new'] = True
            if st.session_state.get('confirm_new'):
                st.warning("Are you sure you want to create a new topic?")
                col1, col2= st.columns(2)
                with col1:
                    if st.button("Yes, create new topic", key=f"create_new_{radio_key}"):
                        st.session_state['confirm_new'] = False
                        saved_ids = [t.get('topic') for t in st.session_state.topicsbert['topics'] if 'topic' in t]
                        next_subtopic_id = next_topic_id(saved_ids, start = 0)
                        new_topic = {
                            'topic': next_subtopic_id,
                            'name': topic['name'],
                            'keywords': topic['keywords'],
                            'documents': topic['documents'],
                            'source': 'Streamlit'
                        }
                        st.session_state.topicsbert['topics'].append(new_topic)
                        local_path = 'Model_training/topics_BERT.json'
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        with open(local_path, 'w', encoding='utf-8') as f:
                            json.dump(st.session_state.topicsbert, f, ensure_ascii=False, indent=2)
                        resp = push_file_to_github('Model_training/topics_BERT.json', repo = 'ERSRisk/tulane-sentiment-app-clean',
                                                              dest_path = 'Model_training/topics_BERT.json', branch = 'main')
                        resp5 = push_file_to_github('Model_training/topics_BERT.json', repo = 'ERSRisk/Tulane-Sentiment-Analysis',
                                                               dest_path = 'Model_training/topics_BERT.json', branch = 'main')
                        unmatched_json = [t for t in st.session_state.unmatched if t['topic'] != topic['topic']]
                        st.session_state.unmatched = unmatched_json
                        saved_topicandsubtopic = [t.get('main_topic_id') for t in st.session_state.topicandsubtopic if 'main_topic_id' in t]
                        new_main_id = next_topic_id(saved_topicandsubtopic, start = 1)
                        new_topicandsubtopic = {
                            'main_topic_id':new_main_id,
                            'main_label':topic['name'],
                            'subtopics':[{
                                'topic_id':next_subtopic_id,
                                'label': topic['name']
                            }]
                        }
                        st.session_state.topicandsubtopic.append(new_topicandsubtopic)
                        auto_path = 'Model_training/topics_BERT_auto.json'
                        os.makedirs(os.path.dirname(auto_path), exist_ok = True)
                        with open(auto_path, 'w', encoding = 'utf-8') as f:
                            json.dump(st.session_state.topicandsubtopic, f, ensure_ascii = False, indent = 2)
                        resp10 = push_file_to_github('Model_training/topics_BERT_auto.json', repo = 'ERSRisk/tulane-sentiment-app-clean',
                                                              dest_path = 'Model_training/topics_BERT_auto.json', branch = 'main')

                        # Update the single canonical unmatched file (no Contents API!)
                        resp6 = upsert_single_big_json(
                            owner="ERSRisk",
                            repo="tulane-sentiment-app-clean",
                            tag="unmatched-topics",
                            asset_name="unmatched_topics.json",
                            new_items=st.session_state.unmatched,
                            dedupe_key="topic",
                            token=os.getenv('GITHUB_TOKEN'),
                            mode = 'replace'
                        )
                        st.success(f"New topic {topic['topic']} created successfully!")
                with col2:
                    if st.button("Cancel", key=f"cancel_new_{radio_key}"):
                        st.session_state['confirm_new'] = False
                        st.session_state[reset_flag] = True
                        st.rerun()
        if decision == 'Merge with existing topic':
            st.session_state['confirm_merge'] = True
            if st.session_state.get('confirm_merge'):
                st.warning("Are you sure you want to merge this topic with an existing one?")
                col1, col2= st.columns(2)
                id_to_name = {int(t['topic']): str(t.get('name', '')).strip()
                             for t in st.session_state.topicsbert['topics'] if 'topic' in t}
                
                for main in st.session_state.topicandsubtopic:
                    for stp in main.get('subtopics', []) or []:
                        tid = int(stp.get('topic_id', -1))
                        if tid in id_to_name:
                            stp['label'] = id_to_name[tid]
                                
                
                main_options = [('--Select a topic--', None)] + [(item['main_label'], item['main_topic_id']) for item in st.session_state.topicandsubtopic]
                
                main_choice = st.selectbox("Select existing topic to merge with:", main_options,index = 0, key=f"existing_topic_{radio_key}")
                selected_main_label, selected_main_id = main_choice if main_choice else (None, None)
                subtopic_list = []
                if selected_main_id is not None:
                    for item in st.session_state.topicandsubtopic:
                        if item["main_topic_id"] == selected_main_id:
                            subtopic_list = item.get("subtopics", []) or []
                            break
                    
                selected_subtopic = None
                if subtopic_list:
                    # Build subtopic options as (label, id) pairs
                    sub_options = [(stp["topic_id"], stp["label"]) for stp in subtopic_list]
                    opts = [(None, '--All Subtopics--')] + sub_options
                    sub_choice = st.selectbox(
                        "Subtopic (optional)",
                        options=opts,
                        format_func=lambda opt: opt[1],
                        index=0,
                        key=f"subtopic_select_{radio_key}",
                    )
                    selected_subtopic = sub_choice[0]  # None means "All subtopics"
                else:
                    st.caption("No subtopics available for this main topic.")
                with col1:
                    if st.button("Yes, merge topic", key=f"merge_{radio_key}"):
                        if selected_subtopic is not None:
                        
                            st.session_state['confirm_merge'] = False
                            
                            for t in st.session_state.topicsbert['topics']:
                                if int(t['topic']) == selected_subtopic:
                                    if isinstance(t['documents'], str):
                                        t['documents'] = [t['documents']]
                                    t['documents'].extend(topic['documents'])
    
                                # Ensure keywords are lists
                                    t['keywords'] = [str(k).strip() for k in t['keywords'] if str(k).strip()]
                                    new_keywords = [k.strip() for k in topic['keywords'].split(',')] if isinstance(topic['keywords'], str) else topic['keywords']
                                    t['keywords'].extend(new_keywords)
                                    with open('Model_training/topics_BERT.json', 'w', encoding='utf-8') as f:
                                        json.dump(st.session_state.topicsbert, f, ensure_ascii=False, indent=2)
                                    resp1 = push_file_to_github('Model_training/topics_BERT.json', repo = 'ERSRisk/tulane-sentiment-app-clean',
                                                                  dest_path = 'Model_training/topics_BERT.json', branch = 'main')
                                    unmatched_json = [t for t in st.session_state.unmatched if t['topic'] != topic['topic']]
                                    st.session_state.unmatched = unmatched_json
    
                                    # Update the single canonical unmatched file (no Contents API!)
                                    resp6 = upsert_single_big_json(
                                        owner="ERSRisk",
                                        repo="tulane-sentiment-app-clean",
                                        tag="unmatched-topics",
                                        asset_name="unmatched_topics.json",
                                        new_items=st.session_state.unmatched,
                                        dedupe_key="topic",
                                        token=os.getenv('GITHUB_TOKEN'),
                                        mode = 'replace'
                                    )
                                    
                                    st.success(f"Topic {topic['topic']} merged successfully!")
                        if selected_subtopic is None:
                            
                            st.session_state['confirm_new'] = False
                            saved_ids = [t.get('topic') for t in st.session_state.topicsbert['topics'] if 'topic' in t]
                            next_subtopic_id = next_topic_id(saved_ids, start = 0)
                            new_topic = {
                                'topic': next_subtopic_id,
                                'name': topic['name'],
                                'keywords': topic['keywords'],
                                'documents': topic['documents'],
                                'source': 'Streamlit'
                            }
                            st.session_state.topicsbert['topics'].append(new_topic)
                            local_path = 'Model_training/topics_BERT.json'
                            os.makedirs(os.path.dirname(local_path), exist_ok=True)
                            with open(local_path, 'w', encoding='utf-8') as f:
                                json.dump(st.session_state.topicsbert, f, ensure_ascii=False, indent=2)
                            push_file_to_github('Model_training/topics_BERT.json', repo = 'ERSRisk/tulane-sentiment-app-clean',
                                                                  dest_path = 'Model_training/topics_BERT.json', branch = 'main')
                            push_file_to_github('Model_training/topics_BERT.json', repo = 'ERSRisk/Tulane-Sentiment-Analysis',
                                                                   dest_path = 'Model_training/topics_BERT.json', branch = 'main') 
                            for main in st.session_state.topicandsubtopic:
                                if main['main_topic_id'] == selected_main_id:
                                    main.setdefault('subtopics', []).append({
                                        'topic_id': next_subtopic_id,
                                        'label': topic['name']
                                    })
                                    break
                            os.makedirs(os.path.dirname('Model_training/topics_BERT_auto.json'), exist_ok=True)
                            with open('Model_training/topics_BERT_auto.json', 'w', encoding = 'utf-8') as f:
                                json.dump(st.session_state.topicandsubtopic, f, ensure_ascii = False, indent = 2)
                            push_file_to_github(
                                'Model_training/topics_BERT_auto.json', repo = 'ERSRisk/tulane-sentiment-app-clean',
                                dest_path = 'Model_training/topics_BERT_auto.json', branch = 'main')
                            
                            unmatched_json = [t for t in st.session_state.unmatched if t['topic'] != topic['topic']]
                            st.session_state.unmatched = unmatched_json
    
                            # Update the single canonical unmatched file (no Contents API!)
                            upsert_single_big_json(
                                owner="ERSRisk",
                                repo="tulane-sentiment-app-clean",
                                tag="unmatched-topics",
                                asset_name="unmatched_topics.json",
                                new_items=st.session_state.unmatched,
                                dedupe_key="topic",
                                token=os.getenv('GITHUB_TOKEN'),
                                mode = 'replace'
                            )
                            
                            st.success(f"Topic {topic['topic']} merged successfully!")
                            st.session_state['confirm_merge'] = False
                        
    
                with col2:
                    if st.button("Cancel", key=f"cancel_merge_{radio_key}"):
                        st.session_state['confirm_merge'] = False
                        st.session_state[reset_flag] = True
                        st.rerun()
        if decision == 'Discard':
            st.session_state[reset_flag] = True
            st.session_state[skip_key] = True

            st.warning(f"Topic {topic['topic']} discarded.")

            discarded_topic = {
                'topic': topic['topic'],
                'name': topic['name'],
                'keywords': topic['keywords'],
                'documents': topic['documents']
            }
            st.session_state.discarded.append(discarded_topic)
            resp2 = upsert_single_big_json(
                    owner= "ERSRisk",
                    repo =  'tulane-sentiment-app-clean',
                    tag="discarded-topics",                     # release to hold the ONE file
                    asset_name="discarded_topics.json",      # SINGLE canonical asset name
                    new_items=st.session_state.discarded,       # your new/edited items
                    dedupe_key="topic",
                    token=os.getenv('GITHUB_TOKEN')
                )

            unmatched_json = [t for t in st.session_state.unmatched if t['topic'] != topic['topic']]
            st.session_state.unmatched = unmatched_json

            # Update the single canonical unmatched file (no Contents API!)
            resp3 = upsert_single_big_json(
                owner="ERSRisk",
                repo="tulane-sentiment-app-clean",
                tag="unmatched-topics",
                asset_name="unmatched_topics.json",
                new_items=st.session_state.unmatched,
                dedupe_key="topic",
                token=os.getenv('GITHUB_TOKEN'),
                mode = 'replace'
            )

            st.success(f"Topic {topic['topic']} discarded successfully!")

if selection == "Article Risk Review":
    import streamlit as st
    import pandas as pd
    import json
    from datetime import datetime
    from datetime import timedelta
    import os
    from pathlib import Path
    import ast
    OWNER = 'ERSRisk'
    REPO = 'Tulane-Sentiment-Analysis'
    TAG = 'BERTopic_results'
    ASSET = 'BERTopic_Streamlit.csv.gz'
    numeric_cols = ['Recency','Source_Accuracy','Impact_Score','Acceleration_value',
                'Location','Industry_Risk','Frequency_Score','Risk_Score','Probability']
    hidden_file = Path('Model_training/hidden_topics.json')

    def atomic_write_json(path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        with open(tmp, 'w', encoding = 'utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @st.cache_data
    def load_hidden_topics(path:str, file_mtime:float) -> list[int]:
        p = Path(path)
        if not p.exists():
            return[]
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [int(t) for t in (data.get('hidden_topic_ids', []) if isinstance(data, dict) else data)]

    def save_hidden_topics(hidden_ids:set[int]):
        payload = {'hidden_topic_ids': sorted(int(x) for x in hidden_ids)}
        atomic_write_json(hidden_file, payload)
        load_hidden_topics.clear()

    hidden_mtime = hidden_file.stat().st_mtime if hidden_file.exists() else 0.0
    hidden_topic_ids = set(load_hidden_topics(str(hidden_file), hidden_mtime))
    @st.cache_data(show_spinner=True, ttl=1800)
    def get_csv_from_release(owner, repo, tag, asset, usecols=None) -> pd.DataFrame:
        token = _github_token()
        if not token:
            raise RuntimeError("GITHUB_TOKEN missing (not injected or empty).")

        headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
        }
        rel = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}",
        headers=headers, timeout=60
        )
        if rel.status_code != 200:
        # show the real reason (401, 404, permissions)
            raise RuntimeError(f"Release lookup {rel.status_code}: {rel.text[:300]}")

        rel_json = rel.json()
        asset_obj = next((a for a in rel_json.get('assets', []) if a.get('name') == asset), None)
        if not asset_obj:
            raise RuntimeError(f"Asset '{asset}' not found in release '{tag}'.")

        url = asset_obj['browser_download_url']
        r = requests.get(url, headers={"Authorization": f"token {token}", "Accept": "application/octet-stream"}, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Asset download {r.status_code}: {r.text[:300]}")
        return pd.read_csv(io.BytesIO(r.content), compression="gzip", low_memory=False, dtype=str, usecols=usecols)
    def get_csv_from_repo(OWNER, REPO, path):
        token = _github_token()
        headers = {
        "Accept": "application/vnd.github.raw",
        "Authorization": f"token {token}",
        }
        api_url = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/main/{path}"
        r= requests.get(api_url, headers = headers)
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content), compression = 'gzip', low_memory = False)
    @st.cache_data
    def load_subtopic_to_main_label_map(auto_path: str = "Model_training/topics_BERT_auto.json") -> dict[int, str]:
        try:
            with open(auto_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}
    
        sub_to_main: dict[int, str] = {}
        if isinstance(data, list):
            # expected shape: [{"main_topic_id": int, "main_label": str, "subtopics":[{"topic_id": int, "label": str}, ...]}, ...]
            for main in data:
                main_label = str(main.get("main_label", "")).strip() or "Unlabeled Topic"
                for stp in (main.get("subtopics") or []):
                    try:
                        tid = int(stp.get("topic_id"))
                        sub_to_main[tid] = main_label
                    except Exception:
                        continue
        return sub_to_main

    required_keys = {'Title', 'Content'}
    if 'articles' not in st.session_state:
        usecols = ['Title', 'Content', 'Link', 'Published', 'University Label', 'Predicted_Risks_new', 'Recency', 'Source_Accuracy',
                  'Impact_Score', 'Acceleration_value', 'Location', 'Industry_Risk', 'Frequency_Score', 'Risk_Score', 'Topic', 'Probability', 'Assigned_how']
        try:
            results_df = load_csv_gz_from_gcs(
                'latest/topics/BERTopic_Streamlit.csv.gz',
                'pipeline/resources/BERTopic_Streamlit.csv.gz')
            st.session_state.articles = results_df.copy()
            #results_df = get_csv_from_release(OWNER, REPO, TAG, ASSET)
        except Exception as e:
            st.error(f"Failed to load BERTopic results: {e}")
            st.stop()
        numeric_cols = ['Recency', 'Source_Accuracy', 'Impact_Score', 'Acceleration_value', 'Location', 'Industry_Risk', 'Frequency_Score', 'Risk_Score', 'Probability', 'Assigned_how']
        use_changes = Path('Model_training/BERTopic_changes.csv').is_file() and Path('Model_training/BERTopic_changes.csv').stat().st_size > 0
        changes_df = None

        if use_changes:
            try:
                changes_df = pd.read_csv('Model_training/BERTopic_changes.csv')
                def norm(s: pd.Series) -> pd.Series:
                    return s.astype(str).str.replace(r'\s+', ' ', regex = True).str.strip()
                for df in (changes_df, results_df):
                    if 'Link' in df.columns:
                        df['Link'] = df['Link'].astype(str).str.strip()
                        df['Title'] = norm(df['Title'])
                        df['Content'] = norm(df['Content'])
                if 'Reviewed' not in changes_df.columns:
                    changes_df['Reviewed'] = 0
                    changes_df['Reviewed'] = pd.to_numeric(changes_df['Reviewed'], errors='coerce').fillna(0).astype(int)


                if not changes_df.empty and required_keys.issubset(changes_df.columns):
                    if 'Changed_at' in changes_df.columns:
                        changes_df['Changed_at'] = pd.to_datetime(changes_df['Changed_at'], errors = 'coerce')
                    if 'Reviewed' not in changes_df.columns:
                        changes_df['Reviewed'] = 0
                    if 'Reviewed_at' not in changes_df.columns:
                        changes_df['Reviewed_at'] = pd.NaT

                    join_keys = ['Link'] if 'Link' in results_df.columns and 'Link' in changes_df.columns else ['Title','Link']
                    review_cols = list({*join_keys, 'Reviewed', 'Reviewed_at', 'Changed_at'})
                    agg = {'Reviewed':'max','Reviewed_at':'max','Changed_at':'max'}
                    review_map = (changes_df[review_cols].dropna(subset=join_keys).groupby(join_keys, as_index = False).agg(agg)
                                 .rename(columns = {'Changed_at': 'Last_changed_at'}))
                else:
                    changes_df = None
            except Exception as e:
                changes_df = None
        if changes_df is not None:
            base = results_df.drop_duplicates(subset = join_keys, keep = 'last')
            merged_df = base.merge(review_map, on = join_keys, how = 'left')
            merged_df['Reviewed'] = merged_df['Reviewed'].fillna(0).astype(int)
            st.session_state.articles = merged_df
        else:
            tmp = results_df.copy()
            tmp['Reviewed'] = 0
            tmp['Reviewed_at'] = pd.NaT
            tmp['Last_changed_at'] = pd.NaT
            st.session_state.articles = tmp
    stories = get_csv_from_repo(OWNER, REPO, 'pipeline/resources/dashboard_stories.csv.gz')
    stories = stories[~stories['canonical_title'].str.match(r'^Story \d+', na=False)]
    
    dropdown = get_csv_from_repo(OWNER, REPO, 'pipeline/resources/dashboard_dropdown.csv.gz')
    change_log_path = Path('Model_training') / 'BERTopic_changes.csv'
    change_log_path.parent.mkdir(parents=True, exist_ok = True)
    if "change_log" not in st.session_state:
        if change_log_path.exists():
            st.session_state.change_log = pd.read_csv(change_log_path)
            for col, default in [('Reviewed', 0), ('Reviewed_at', pd.NaT)]:
                if col not in st.session_state.change_log.columns:
                    st.session_state.change_log[col] = default
        else:
            base_cols = list(st.session_state.articles.columns)
            new_cols = ['Recency_Upd', 'Acceleration_value_Upd', 'Source_Accuracy_Upd',
                    'Impact_Score_Upd', 'Location_Upd', 'Industry_Risk_Upd', 'Frequency_Score_Upd',
                    'Change reason']
            st.session_state.change_log = pd.DataFrame(columns = base_cols + new_cols)
            st.session_state.change_log.to_csv(change_log_path, index = False)
    story_change_log = Path('Model_training') / 'Story_changes.csv'
    story_change_log.parent.mkdir(parents = True, exist_ok = True)
    if "story_log" not in st.session_state:
        if story_change_log.exists():
            st.session_state.story_log = pd.read_csv(story_change_log)
            for col, default in [('Reviewed', 0), ('Reviewed_at', pd.NaT)]:
                if col not in st.session_state.story_log.columns:
                    st.session_state.story_log[col] = default
        else:
            base_cols = list(stories.columns)
            new_cols = ['Recency_Upd', 'Acceleration_value_Upd', 'Source_Accuracy_Upd',
                    'Impact_Score_Upd', 'Location_Upd', 'Industry_Risk_Upd', 'Frequency_Score_Upd',
                    'Change reason']
            st.session_state.story_log = pd.DataFrame(columns = base_cols + new_cols)
            st.session_state.story_log.to_csv(story_change_log, index = False)
    for c in numeric_cols:
        if c in st.session_state.articles.columns:
            st.session_state.articles[c] = pd.to_numeric(st.session_state.articles[c], errors = 'coerce')

    
    #canonical_stories =get_csv_from_repo(OWNER, REPO, 'Model_training/Canonical_Stories_with_Summaries.csv')

    dropdown['Published_utc'] = pd.to_datetime(dropdown['Published_utc'], errors = 'coerce', utc=True)

    story_ids = set(stories['story_id'].dropna())
    
    stories_timeline = stories.copy()
    #stories_timeline = stories_timeline.drop(columns = ['canonical_title', 'summary'])
    #stories_timeline = stories_timeline.merge(canonical_stories[['story_id', 'canonical_title', 'summary']], on = 'story_id', how = 'left')
    stories_timeline['item_type'] = 'story'
    stories_timeline['Published'] = pd.to_datetime(stories_timeline['last_seen'], utc=True, infer_datetime_format = True)
    if stories_timeline['Published'].isna().all():
        st.error("All last_seen timestamps failed to parse")
        st.stop()
    stories_timeline['Title'] = stories_timeline['canonical_title']
    stories_timeline['Content'] = stories_timeline['summary']
    stories_timeline['Link'] = None
    stories_timeline['Reviewed'] = pd.NA
    stories_timeline['risk_label'] = stories_timeline.get('risk_label', None)

    ##adding to push changes to the Github repo
    def push_file_to_github(local_path:str, repo:str, dest_path:str, branch:str = "main", token:str|None = None):
        token = os.getenv('GITHUB_TOKEN')

        with open(local_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("utf-8")

        api_base = f"https://api.github.com/repos/{repo}/contents/{dest_path}"
        headers = {"Authorization": f"token {token}", "Accept":"application/vnd.github+json"}

        sha = None
        r_get = requests.get(api_base, headers = headers, params = {"ref":branch})
        if r_get.status_code == 200:
            sha = r_get.json()['sha']
        payload = {
            "message": f"Update {dest_path} via Streamlit at {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "content": content_b64,
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        r_put = requests.put(api_base, headers = headers, data = json.dumps(payload))
        if r_put.status_code not in (200, 201):
            raise RuntimeError(f"GitHub push failed: {r_put.status_code} {r_put.text}")
        return r_put.json()
    st.title("Article Risk Review Portal")
    #give me a filter to filter articles by date range
    st.sidebar.header("Filter Articles")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=30))
    end_date = st.sidebar.date_input("End Date", datetime.now())


    if start_date > end_date:
        st.sidebar.error("Start date must be before end date.")
    # Load articles and risks


    update_cols = ['Recency_Upd', 'Acceleration_value_Upd', 'Source_Accuracy_Upd',
                    'Impact_Score_Upd', 'Location_Upd', 'Industry_Risk_Upd', 'Frequency_Score_Upd',
                    'Change reason']
    for col in update_cols:
        if col not in st.session_state.articles.columns:
            st.session_state.articles[col] = None



    status_choice = st.sidebar.radio(
        'Review status',
        ['Unreviewed only', 'Reviewed only', 'All'],
        index = 0
    )
    articles_df = st.session_state.articles.copy()
    articles_df = articles_df.drop(columns = 'story_id')
    articles_df = articles_df.merge(
    dropdown[['Link', 'story_id']],
    on='Link',
    how='left'
    )
    
    # Remove articles that belong to a story
    articles_df = articles_df[~articles_df['story_id'].isin(story_ids)]
    articles_df['item_type'] = 'article'
    
    stories_df = stories_timeline.copy()
    stories_df['item_type'] = 'story'
    
    timeline_df = pd.concat(
    [stories_df, articles_df],
    ignore_index=True,
    sort=False
    )

    filtered_df = timeline_df.copy()
    if status_choice == 'Unreviewed only':
        filtered_df = filtered_df[(filtered_df['item_type'] == 'story') | (filtered_df['Reviewed'] != 1)]
    elif status_choice == 'Reviewed only':
        article_only = base_df.merge(last[keys + keep_cols], on=keys, how='inner', suffixes = ('', '_chg'))
        article_only['item_type'] = 'article'
        filtered_df = pd.concat([stories_df, article_only], ignore_index = True, sort = False)
        if 'Reviewed_chg' in filtered_df.columns:
            filtered_df['Reviewed'] = filtered_df['Reviewed_chg'].fillna(filtered_df.get('Reviewed', 0)).astype(int)
            filtered_df.drop(columns = [c for c in ['Reviewed_chg'] if c in filtered_df.columns], inplace = True)

    start_date = pd.to_datetime(start_date).tz_localize(ZoneInfo("America/Chicago")).tz_convert('UTC')
    end_date = (pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)).tz_localize(ZoneInfo("America/Chicago")).tz_convert('UTC')
    filtered_df['Published'] = pd.to_datetime(filtered_df['Published'], errors = 'coerce', utc = True)
    stories_mask = filtered_df['item_type'] == 'story'
    articles_mask = (
        (filtered_df['item_type'] == 'article') &
        (filtered_df['Published'].between(start_date, end_date, inclusive='both'))
    )
    
    filtered_df = filtered_df[stories_mask | articles_mask]
    filtered_df = filtered_df.sort_values('Published', ascending = False, na_position = 'last')
    filtered_df = filtered_df.reset_index(drop = True)

    with open('Model_training/risks.json', 'r') as f:
        risks_data = json.load(f)

    all_possible_risks = [risk['name'] for group in risks_data['new_risks'] for risks in group.values() for risk in risks]
    if "No Risk" not in all_possible_risks:
        all_possible_risks.append("No Risk")
    all_possible_risks = [r for r in all_possible_risks if isinstance(r, str)]
    filter_risks = all_possible_risks[:]

    filtered_risks = st.multiselect("Select Risks to Filter Articles", options = all_possible_risks, default=filter_risks, key="risk_filter")

    def match_any(predicted, selected):
        if not isinstance(predicted, list) or not predicted:
        # Treat empty as "No Risk"
            return "no risk" in selected
        predicted = [str(p).strip().lower() for p in predicted if isinstance(p, str)]
        selected = [s.strip().lower() for s in selected]
        return any(p in selected for p in predicted)
    filtered_df["Link_norm"] = filtered_df["Link"].astype(str).str.strip()
    filtered_df.loc[filtered_df["Link"].isna(), "Link_norm"] = ""
    
    filtered_df["dedupe_key"] = None
    is_story = filtered_df["item_type"].eq("story")
    is_article = filtered_df["item_type"].eq("article")
    
    filtered_df.loc[is_story, "dedupe_key"] = "story:" + filtered_df.loc[is_story, "story_id"].astype(str)
    
    filtered_df.loc[is_article, "dedupe_key"] = (
        "article:" +
        filtered_df.loc[is_article, "Link_norm"].where(
            filtered_df.loc[is_article, "Link_norm"] != "",
            filtered_df.loc[is_article, "Title"].astype(str) + "|" +
            filtered_df.loc[is_article, "Published"].astype(str)
        )
    )
    
    filtered_df = filtered_df.drop_duplicates(subset=["dedupe_key"], keep="last")

    def normalize_risk(x):
        if pd.isna(x):
            return "No Risk"
    
        s = str(x).strip().lower()
    
        bad = {
            '',
            'no risk',
            'none',
            'nan',
            '[]',
            "['no risk']"
        }
    
        return "No Risk" if s in bad else s
    
    filtered_df['Risk_Normalized'] = (
        filtered_df['Predicted_Risks_new']
        .apply(normalize_risk)
    )
    
    filtered_df = filtered_df[
        (filtered_df['item_type'] == 'story') |
        (filtered_df['Risk_Normalized'] != 'No Risk')
    ]

    PAGE_SIZE = st.sidebar.selectbox('Items per Page', [10, 20, 30, 50], index =1)
    total = len(filtered_df)
    max_page = max(1, (total + PAGE_SIZE - 1)//PAGE_SIZE)
    story_positions = np.flatnonzero(filtered_df['item_type'].eq('story').to_numpy()).tolist()
    st.sidebar.write("First story position:", story_positions[0] if story_positions else None)
    if story_positions:
        first_story_page = (story_positions[0] // PAGE_SIZE) + 1
        st.sidebar.write("First story is on page:", first_story_page)

    if 'page_num' not in st.session_state:
        st.session_state.page_num = 1
    st.session_state.page_num = st.sidebar.number_input(
        'Page', min_value = 1, max_value = max_page, value = st.session_state.page_num, step =1
    )

    start = (st.session_state.page_num - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    st.caption(f"Showing {start + 1} to {min(end, total)} of {total} articles")
    page_df = filtered_df.iloc[start:end]

    with open('Model_training/topics_BERT.json', 'r', encoding='utf-8') as f:
        name_map = {int(t['topic']): t['name'] for t in json.load(f)['topics']}
    
    # NEW:
    subtopic_to_main = load_subtopic_to_main_label_map("Model_training/topics_BERT_auto.json")

    hidden_names = [f"{tid} - {name_map.get(tid, 'Unlabeled Topic')}" for tid in sorted(hidden_topic_ids)]
    st.sidebar.markdown('Hidden topics')
    to_unhide = st.sidebar.multiselect(
        'Unhide selected',
        options = sorted(hidden_topic_ids),
        format_func = lambda tid: f"{tid} - {name_map.get(tid, 'Unlabeled Topic')}"
    )
    cola, colb = st.sidebar.columns(2)
    with cola:
        if st.button('Unhide', key='unhide_btn', disabled = not to_unhide):
            hidden_topic_ids.difference_update(to_unhide)
            save_hidden_topics(hidden_topic_ids)
            st.rerun()

    def coerce_topic_scalar(x):
        v = pd.to_numeric(x, errors = 'coerce')
        if pd.isna(v):
            return -1
        try:
            return int(v)
        except Exception as e:
            return e

    articles_by_story = {
                    k: v.sort_values('Published_utc', ascending = False) for k, v in dropdown.groupby('story_id')
                }
    if filtered_risks:
        mask = (
            (filtered_df["item_type"] == "story") |
            filtered_df["Predicted_Risks_new"].apply(
                lambda x: match_any(x, filtered_risks)
            )
        )
        filtered_df = filtered_df[mask]
    rendered_anything = False
    for _, article in page_df.iterrows():
        if article.get('item_type') == 'story':
            
            rendered_anything = True
            story = article
            raw = story.get("Predicted_Risks_new", "[]")

            predicted = []
            
            if isinstance(raw, list):
                predicted = [str(x).strip() for x in raw if str(x).strip()]
            elif isinstance(raw, str):
                s = raw.strip()
                try:
                    predicted = json.loads(s) if s.startswith('[') else [s]
                except Exception:
                    predicted = [s]
            
            if not predicted:
                predicted = ["No Risk"]
            title = str(story.get('Title')).strip()
            if not title or title.lower() == 'nan':
                title = f"Story {story['story_id']}"
            
            with st.expander(title):
                st.markdown(story['Content'])
                w = {
                'Recency': 0.15,
                'Source_Accuracy': 0.10,
                'Impact_Score': 0.35,
                'Acceleration_value': 0.25,
                'Location': 0.05,
                'Industry_Risk': 0.05,
                'Frequency_Score': 0.05
                }
                weight_sum = sum(w.values())
        
                num = (
                    float(story['avg_recency']) * w['Recency'] +
                    float(story['avg_source_accuracy']) * w['Source_Accuracy'] +
                    float(story['avg_impact_score']) * w['Impact_Score'] +
                    float(story['avg_acceleration']) * w['Acceleration_value'] +
                    float(story['avg_location']) * w['Location'] +
                    float(story['avg_industry_risk']) * w['Industry_Risk'] +
                    float(story['avg_frequency']) * w['Frequency_Score']
                )
                story['Risk_Score_y'] = (num / weight_sum)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('**Risk Score** ')
                    st.markdown(story['Risk_Score_y'])
        
                # --- Quick review toggle ---
        
                st.markdown("**Predicted Risks:** " + ((story['risk_label']) if story['risk_label'] else "No Risk"))
        
                tab1, tab2, tab3 = st.tabs(['View Risk Labels', 'Manually Update Risk Labels', 'View Articles'])
                with tab1:
                    col1, col2, col3, col4, col5, col6, col7 =  st.columns(7)
                    with col1:
                        st.metric('Recency', story['avg_recency'])
                    with col2:
                        st.metric('Acceleration', story['avg_acceleration'])
                    with col3:
                        st.metric('Source Accuracy', story['avg_source_accuracy'])
                    with col4:
                        st.metric('Impact Score', story['avg_impact_score'])
                    with col5:
                        st.metric('Location', story['avg_location'])
                    with col6:
                        st.metric('Industry Risk', story['avg_industry_risk'])
                    with col7:
                        st.metric('Frequency', story['avg_frequency'])
                with tab3:
                    sid = story['story_id']
                    if sid not in articles_by_story:
                        st.info("No articles found for this story.")
                    else:
                        articles = articles_by_story[sid]
        
                        st.markdown(f"**{len(articles)} articles found for this story:**")
        
                        for _, art in articles.iterrows():
                            st.markdown('---')
                            st.markdown(f"### {art['Title']}")
                            st.markdown(f"**Published:** {art['Published_utc']}")
                            st.markdown(f"**Source:** {art.get('Source', 'Unknown')}")
                            st.markdown(f"**Link:** {art['Link']}")
                            if 'Risk_Score' in art:
                                st.markdown("---")
                                st.markdown("**Article Signals**")
                                st.caption(
                                    f"""
                                    Risk: {art.get('Risk_Score', '—')} | 
                                    Recency: {art.get('Recency', '—')} | 
                                    Acceleration: {art.get('Acceleration_value', '—')} | 
                                    Frequency: {art.get('Frequency_Score', '—')}
                                    """
                                )  
                with tab2:
                    options = [0.0, 1.0,2.0,3.0,4.0,5.0]
                    with st.form(f"manual_edit_form_{story['story_id']}"):
                        raw = risks_data.get('new_risks', risks_data) if isinstance(risks_data, dict) else risks_data
                        categories = {}
                        if isinstance(raw, list):
                            for item in raw:
                                if not isinstance(item, dict):
                                    continue
                                for cat, entries in item.items():
                                    names = []
                                    for entry in entries:
                                        if isinstance(entry, dict) and 'name' in entry:
                                            names.append(str(entry['name']))
                                        elif isinstance(entry, str):
                                            names.append(entry)
                                    if names:
                                        categories[str(cat)] = names
                        else:
                            st.error('risks.json format unexpected : new_risks is not a list')
                            categories = {}
                        pairs = [(cat, risk_name) for cat, lst in categories.items() for risk_name in lst]

                        if all(risk != 'No Risk' for _, risk in pairs):
                            pairs.append(('General', 'No Risk'))
                        if not pairs:
                            st.warning('No risks loaded.')
                            selected_risks = []
                        else:
                            pred_set = {str(p).strip().lower() for p in predicted if isinstance(p, str)}
                            default_pair = next((pr for pr in pairs if pr[1].strip().lower() in pred_set), None)
                            default_index = pairs.index(default_pair) if default_pair in pairs else 0
                           #valid_defaults = [opt for opt in all_possible_risks if any(opt.lower() == str(p).lower() for p in predicted if isinstance(p, str))]
                            #selected_risks = st.multiselect(
                             #   "Edit risks if necessary:",
                              #  options=all_possible_risks,
                               # default=valid_defaults,
                                #key=f"edit_{idx}"
                            #)
                            choice = st.selectbox(
                                "Edit risk if necessary (one selection):",
                                options = pairs,
                                index = default_index,
                                format_func=lambda pr: f"{pr[0]} ▸ {pr[1]}",
                                key = f"edit_c_{story['story_id']}"
                            )
                            selected_risks = [choice[1]]
                        col1, col2, col3, col4, col5, col6, col7 =  st.columns(7)
                        with col1:
                            upd_recency_value = st.number_input('Recency Risk', min_value = 0.0, max_value = 5.0, step = 1.0, value= float(story['Recency_Upd'] if pd.notna(story['Recency_Upd']) else story['avg_recency']), key =f"recency_input_{story['story_id']}")
                        with col2:
                            upd_acceleration_value = st.number_input('Acceleration Risk',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(story['Acceleration_value_Upd'] if pd.notna(story['Acceleration_value_Upd']) else story['avg_acceleration']), key =f"acceleration_input_{story['story_id']}")
                        with col3:
                            upd_source_accuracy =st.number_input('Source Accuracy',  min_value=0.0, max_value = 5.0, step = 1.0, value= float(story['Source_Accuracy_Upd'] if pd.notna(story['Source_Accuracy_Upd']) else story['avg_source_accuracy']), key =f"source_input_{story['story_id']}")
                        with col4:
                            upd_impact_score = st.number_input('Impact Score',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(story['Impact_Score_Upd'] if pd.notna(story['Impact_Score_Upd']) else story['avg_impact_score']), key =f"impact_input_{story['story_id']}")
                        with col5:
                            upd_location=st.number_input('Location Risk',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(story['Location_Upd'] if pd.notna(story['Location_Upd']) else story['avg_location']), key =f"location_input_{story['story_id']}")
                        with col6:
                            upd_industry_risk = st.number_input('Industry Risk',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(story['Industry_Risk_Upd'] if pd.notna(story['Industry_Risk_Upd']) else story['avg_industry_risk']), key =f"industry_input_{story['story_id']}")
                        with col7:
                            upd_frequency_score = st.number_input('Frequency Score', min_value=0.0, max_value = 5.0, step = 1.0, value=float(story['Frequency_Score_Upd'] if pd.notna(story['Frequency_Score_Upd']) else story['avg_frequency']), key =f"frequency_input_{story['story_id']}")

                        st.markdown('Please provide a reason for the changes made to the risk labels:')
                        reason = st.text_area("Reason for changes", placeholder="Explain the changes made to the risk labels.", key=f"reason_{story['story_id']}")
                        submitted =  st.form_submit_button("Update Risk Labels")
                        if submitted:
                            #st.write("This feature is not yet available since this is a story made up of several articles. It will be available within the next couple of days.")
                            full_story_df = pd.read_csv(story_change_log)
                            mask = ((full_story_df['canonical_title'] == story['Title']))
                            if mask.any():
                                base_row = full_story_df.loc[mask].sort_values('Changed_at').iloc[-1].to_dict()
                            else:
                                base_row = story.copy()
                                if hasattr(base_row, 'to_dict'):
                                    base_row = base_row.to_dict()

                            base_row['Predicted_Risks_Upd'] = selected_risks
                            base_row['Recency_Upd'] = upd_recency_value
                            base_row['Acceleration_value_Upd'] = upd_acceleration_value
                            base_row['Source_Accuracy_Upd'] = upd_source_accuracy
                            base_row['Impact_Score_Upd'] = upd_impact_score
                            base_row['Location_Upd'] = upd_location
                            base_row['Industry_Risk_Upd'] = upd_industry_risk
                            base_row['Frequency_Score_Upd'] = upd_frequency_score
                            base_row['Change reason'] = reason
                            base_row['Changed_at'] = pd.Timestamp.utcnow()
                            base_row['Reviewed'] = 1
                            base_row['Reviewed_at'] = pd.Timestamp.utcnow()

                            st.session_state.story_log = pd.concat(
                                [st.session_state.story_log, pd.DataFrame([base_row])],
                                ignore_index=True
                            )
                            expected_cols = set(st.session_state.story_log.columns)
                            row_cols = set(base_row.keys())
                            
                            if row_cols != expected_cols:
                                st.error(f"Schema mismatch. Refusing to save.\nMissing: {expected_cols - row_cols}")
                                st.stop()

                            st.session_state.story_log.to_csv(story_change_log, index = False)
                            try:
                                resp = push_file_to_github(story_change_log, repo = 'ERSRisk/tulane-sentiment-app-clean',
                                                          dest_path = 'Model_training/Story_changes.csv', branch = 'main')
                                changes = pd.read_csv('Model_training/Story_changes.csv')
                                res = pd.read_csv('BERTopic_results.csv')
                                Change_timestamp = 'Changed_at'
                                changes_sorted = changes.sort_values(Change_timestamp).drop_duplicates(['Title', 'Content'], keep = 'last')


                                st.success('Saved changes')
                            except Exception as e:
                                st.error(f"Github failed to push: {e}")
            continue
        
        row_id = hash(article.get('Link', article.get('Title')))
        reviewed = bool(int(article.get('Reviewed', 0)))
        badge = "✅ Reviewed" if reviewed else "Not reviewed"
        title = str(article.get("Title", ""))[:100]
    
    
        raw = article.get("Predicted_Risks_new", "[]")
        
        predicted = []
        
        if isinstance(raw, list):
            predicted = [str(x).strip() for x in raw if str(x).strip()]
        
        elif isinstance(raw, str):
            s = raw.strip()
            predicted = []
        
            # 1) Try JSON
            if s.startswith('[') and s.endswith(']'):
                try:
                    j = json.loads(s)
                    if isinstance(j, list):
                        predicted = [str(x).strip() for x in j if str(x).strip()]
                except Exception:
                    try:
                        import ast
                        j = ast.literal_eval(s)
                        if isinstance(j, list):
                            predicted = [str(x).strip() for x in j if str(x).strip()]
                    except Exception:
                        pass
        
            # 3) fallback parsing
            if not predicted and s:
                sep = ';' if ';' in s else (',' if ',' in s else None)
                if sep:
                    predicted = [p.strip() for p in s.split(sep) if p.strip()]
                else:
                    predicted = [s]
        
            # 4) normalize
            if not predicted or all(p.lower() in ("no risk", "none") for p in predicted):
                predicted = ["No Risk"]
        else:
            predicted = raw
           
            

        if article['item_type'] == 'article':
            title = str(article.get("Title", ""))[:100]
        
            if title:
        
                tid = coerce_topic_scalar(article.get('Topic'))
                article['Topic'] = tid
        
                article['Topic_name'] = subtopic_to_main.get(tid, name_map.get(tid, 'Unlabeled Topic'))
                if tid in hidden_topic_ids:
                    continue
                
            
                with st.expander(f"{badge} — {title}..."):
                    st.markdown(f"[Read full article]({article['Link']})")
                    st.write(article['Content'][:1000])
                    w = {
                    'Recency': 0.15,
                    'Source_Accuracy': 0.10,
                    'Impact_Score': 0.35,
                    'Acceleration_value': 0.25,
                    'Location': 0.05,
                    'Industry_Risk': 0.05,
                    'Frequency_Score': 0.05
                    }
                    weight_sum = sum(w.values())
                    num = (
                        float(article['Recency']) * w['Recency'] +
                        float(article['Source_Accuracy']) * w['Source_Accuracy'] +
                        float(article['Impact_Score']) * w['Impact_Score'] +
                        float(article['Acceleration_value']) * w['Acceleration_value'] +
                        float(article['Location']) * w['Location'] +
                        float(article['Industry_Risk']) * w['Industry_Risk'] +
                        float(article['Frequency_Score']) * w['Frequency_Score']
                    )
                    article['Risk_Score_y'] = (num / weight_sum)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('**Risk Score** ')
                        st.markdown(article['Risk_Score_y'])
                    with col2:
                        st.markdown(article['Assigned_how'])
                    c1, c2 = st.columns(2)
                    with c1:
                        if not reviewed:
                            row_id = hash(article.get('Link'))
                            if st.button("Mark as reviewed", key=f"mark_{row_id}"):
                                new_row = article.to_dict()
                                new_row['Reviewed'] = 1
                                new_row['Reviewed_at'] = pd.Timestamp.utcnow()
                                new_row['Changed_at'] = new_row.get('Changed_at', pd.Timestamp.utcnow())
                                st.session_state.change_log = pd.concat(
                                    [st.session_state.change_log, pd.DataFrame([new_row])],
                                    ignore_index=True
                                )
                                st.session_state.change_log.to_csv(change_log_path, index=False)
                                st.success("Marked reviewed ✅")
                                st.rerun()
                        else:
                            if st.button("Unmark reviewed", key=f"unmark_{row_id}"):
                                new_row = article.to_dict()
                                new_row['Reviewed'] = 0
                                new_row['Reviewed_at'] = pd.NaT
                                new_row['Changed_at'] = new_row.get('Changed_at', pd.Timestamp.utcnow())
                                st.session_state.change_log = pd.concat(
                                    [st.session_state.change_log, pd.DataFrame([new_row])],
                                    ignore_index=True
                                )
                                st.session_state.change_log.to_csv(change_log_path, index=False)
                                st.info("Review mark removed")
                                st.rerun()
                    with c2:
                        if st.button('Hide this topic', key = f'hide_topic_{tid}_{row_id}'):
                            if tid != -1:
                                    hidden_topic_ids.add(int(tid))
                                    save_hidden_topics(hidden_topic_ids)
                                    st.success(f"Hid topic {tid} - {article['Topic_name']}")
                                    st.rerun()
        
                        shown = [str(p) for p in predicted if str(p).strip()]
        
                        st.markdown("**Predicted Risks:** " + (", ".join(shown) if shown else "No Risk"))
        
        
                    tab1, tab2 = st.tabs(['View Risk Labels', 'Manually Update Risk Labels'])
                    with tab1:
                        col1, col2, col3, col4, col5, col6, col7 =  st.columns(7)
                        with col1:
                            st.metric('Recency', article['Recency_Upd'] if pd.notna(article['Recency_Upd']) else article['Recency'])
                        with col2:
                            st.metric('Acceleration', article['Acceleration_value_Upd'] if pd.notna(article['Acceleration_value_Upd']) else article['Acceleration_value'])
                        with col3:
                            st.metric('Source Accuracy', article['Source_Accuracy_Upd'] if pd.notna(article['Source_Accuracy_Upd']) else article['Source_Accuracy'])
                        with col4:
                            st.metric('Impact Score', article['Impact_Score_Upd'] if pd.notna(article['Impact_Score_Upd']) else article['Impact_Score'])
                        with col5:
                            st.metric('Location', article['Location_Upd'] if pd.notna(article['Location_Upd']) else article['Location'])
                        with col6:
                            st.metric('Industry Risk', article['Industry_Risk_Upd'] if pd.notna(article['Industry_Risk_Upd']) else article['Industry_Risk'])
                        with col7:
                            st.metric('Frequency', article['Frequency_Score_Upd'] if pd.notna(article['Frequency_Score_Upd']) else article['Frequency_Score'])
#    
                        with tab2:
                            options = [0.0, 1.0,2.0,3.0,4.0,5.0]
                            with st.form(f"manual_edit_form_{row_id}"):
                                raw = risks_data.get('new_risks', risks_data) if isinstance(risks_data, dict) else risks_data
                                categories = {}
                                if isinstance(raw, list):
                                    for item in raw:
                                        if not isinstance(item, dict):
                                            continue
                                        for cat, entries in item.items():
                                            names = []
                                            for entry in entries:
                                                if isinstance(entry, dict) and 'name' in entry:
                                                    names.append(str(entry['name']))
                                                elif isinstance(entry, str):
                                                    names.append(entry)
                                            if names:
                                                categories[str(cat)] = names
                                else:
                                    st.error('risks.json format unexpected : new_risks is not a list')
                                    categories = {}
                                pairs = [(cat, risk_name) for cat, lst in categories.items() for risk_name in lst]
    
                                if all(risk != 'No Risk' for _, risk in pairs):
                                    pairs.append(('General', 'No Risk'))
                                if not pairs:
                                    st.warning('No risks loaded.')
                                    selected_risks = []
                                else:
                                    pred_set = {str(p).strip().lower() for p in predicted if isinstance(p, str)}
                                    default_pair = next((pr for pr in pairs if pr[1].strip().lower() in pred_set), None)
                                    default_index = pairs.index(default_pair) if default_pair in pairs else 0
                                   #valid_defaults = [opt for opt in all_possible_risks if any(opt.lower() == str(p).lower() for p in predicted if isinstance(p, str))]
                                    #selected_risks = st.multiselect(
                                     #   "Edit risks if necessary:",
                                      #  options=all_possible_risks,
                                       # default=valid_defaults,
                                        #key=f"edit_{idx}"
                                    #)
                                    choice = st.selectbox(
                                        "Edit risk if necessary (one selection):",
                                        options = pairs,
                                        index = default_index,
                                        format_func=lambda pr: f"{pr[0]} ▸ {pr[1]}",
                                        key = f"edit_c_{row_id}"
                                    )
                                    selected_risks = [choice[1]]
                                col1, col2, col3, col4, col5, col6, col7 =  st.columns(7)
                                with col1:
                                    upd_recency_value = st.number_input('Recency Risk', min_value = 0.0, max_value = 5.0, step = 1.0, value= float(article['Recency_Upd'] if pd.notna(article['Recency_Upd']) else article['Recency']), key =f"recency_input_{row_id}")
                                with col2:
                                    upd_acceleration_value = st.number_input('Acceleration Risk',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(article['Acceleration_value_Upd'] if pd.notna(article['Acceleration_value_Upd']) else article['Acceleration_value']),key =f"acceleration_input_{row_id}")
                                with col3:
                                    upd_source_accuracy =st.number_input('Source Accuracy',  min_value=0.0, max_value = 5.0, step = 1.0, value= float(article['Source_Accuracy_Upd'] if pd.notna(article['Source_Accuracy_Upd']) else article['Source_Accuracy']),key =f"source_input_{row_id}")
                                with col4:
                                    upd_impact_score = st.number_input('Impact Score',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(article['Impact_Score_Upd'] if pd.notna(article['Impact_Score_Upd']) else article['Impact_Score']),key =f"impact_input_{row_id}")
                                with col5:
                                    upd_location=st.number_input('Location Risk',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(article['Location_Upd'] if pd.notna(article['Location_Upd']) else article['Location']),key =f"location_input_{row_id}")
                                with col6:
                                    upd_industry_risk = st.number_input('Industry Risk',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(article['Industry_Risk_Upd'] if pd.notna(article['Industry_Risk_Upd']) else article['Industry_Risk']),key =f"industry_input_{row_id}")
                                with col7:
                                    upd_frequency_score = st.number_input('Frequency Score', min_value=0.0, max_value = 5.0, step = 1.0, value=float(article['Frequency_Score_Upd'] if pd.notna(article['Frequency_Score_Upd']) else article['Frequency_Score']),key =f"frequency_input_{row_id}")
    
                                st.markdown('Please provide a reason for the changes made to the risk labels:')
                                reason = st.text_area("Reason for changes", placeholder="Explain the changes made to the risk labels.", key=f"reason_{row_id}")
                                submitted =  st.form_submit_button("Update Risk Labels")
                                if submitted:
                                    new_row = article.copy()
                                    new_row = new_row.to_dict()
    
                                    new_row['Predicted_Risks_Upd'] = selected_risks
                                    new_row['Recency_Upd'] = upd_recency_value
                                    new_row['Acceleration_value_Upd'] = upd_acceleration_value
                                    new_row['Source_Accuracy_Upd'] = upd_source_accuracy
                                    new_row['Impact_Score_Upd']= upd_impact_score 
                                    new_row['Location_Upd']= upd_location 
                                    new_row['Industry_Risk_Upd'] = upd_industry_risk 
                                    new_row['Frequency_Score_Upd']= upd_frequency_score
                                    new_row['Change reason'] = reason
                                    new_row['Changed_at'] = pd.Timestamp.utcnow().isoformat(timespec = 'seconds')
                                    new_row['Changed_at'] = pd.to_datetime(new_row['Changed_at'], errors = 'coerce')
                                    new_row['Reviewed'] = 1
                                    new_row['Reviewed_at'] = pd.Timestamp.utcnow()
    
                                    st.session_state.change_log = pd.concat(
                                        [st.session_state.change_log, pd.DataFrame([new_row])],
                                        ignore_index = True
                                    )
    
                                    st.session_state.change_log.to_csv(change_log_path, index = False)
                                    try:
                                        resp = push_file_to_github(change_log_path, repo = 'ERSRisk/tulane-sentiment-app-clean',
                                                                  dest_path = 'Model_training/BERTopic_changes.csv', branch = 'main')
                                        changes = pd.read_csv('Model_training/BERTopic_changes.csv')
                                        res = pd.read_csv('BERTopic_results.csv')
                                        Change_timestamp = 'Changed_at'
                                        changes_sorted = changes.sort_values(Change_timestamp).drop_duplicates(['Title', 'Content'], keep = 'last')
   
    
                                        st.success('Saved changes')
                                    except Exception as e:
                                        st.error(f"Github failed to push: {e}")
   
if selection == "Risk/Event Detector":
    import streamlit as st
    import pdfplumber
    import docx
    import re
    import json
    from collections import defaultdict
    from sentence_transformers import SentenceTransformer, util
    import pandas as pd
    import altair as alt

    st.title("📄 Risk/Event Detector & Trend Analysis")

    # --- Load risk definitions ---
    @st.cache_data
    def load_risk_definitions():
        with open("Model_training/risks.json", "r") as f:
            raw_data = json.load(f)
        reformatted = {}
        for category_entry in raw_data["new_risks"]:
            for category, items in category_entry.items():
                reformatted[category] = {
                    "keywords": [item["name"] for item in items],
                    "description": f"Risks in category: {category}"
                }
        return reformatted

    # --- Load BERTopic results ---
    @st.cache_data
    def load_bertopic_results():
        df = pd.read_csv("BERTopic_results.csv")
        df['Published'] = pd.to_datetime(df['Published'], errors='coerce')
        return df.dropna(subset=['Published'])

    bertopic_df = load_bertopic_results()

    # --- Load model ---
    model = SentenceTransformer("all-MiniLM-L6-v2")
    risk_event_definitions = load_risk_definitions()

    # --- Text extraction helpers ---
    def extract_text_from_pdf(file):
        with pdfplumber.open(file) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        return text

    def extract_text_from_docx(file):
        doc = docx.Document(file)
        return '\n'.join([para.text for para in doc.paragraphs])

    def extract_text_from_txt(file):
        return file.read().decode('utf-8')

    # --- Semantic similarity ---
    def extract_semantic_risk_sentences(text, definitions, threshold=0.5):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        risk_labels = list(definitions.keys())
        risk_descriptions = [definitions[label]["description"] for label in risk_labels]
        risk_embeddings = model.encode(risk_descriptions, convert_to_tensor=True)
        matches = defaultdict(list)
        for i, sentence in enumerate(sentences):
            scores = util.cos_sim(sentence_embeddings[i], risk_embeddings)[0]
            for j, score in enumerate(scores):
                if score >= threshold:
                    matches[risk_labels[j]].append((sentence.strip(), round(score.item(), 3)))
        return dict(matches)

    # --- Risk trend analysis ---
    def check_risk_trend(risk_label, weeks_window=6):
        df_risk = bertopic_df[bertopic_df['Detected_Risks'] == risk_label]
        if df_risk.empty:
            return None, None, False

        weekly_counts = (
            df_risk.groupby(pd.Grouper(key='Published', freq='W'))
            .size()
            .reset_index(name='mentions')
            .sort_values('Published')
        )

        if len(weekly_counts) < weeks_window:
            return weekly_counts, None, False

        recent_avg = weekly_counts['mentions'].iloc[-weeks_window//2:].mean()
        older_avg = weekly_counts['mentions'].iloc[-weeks_window:].mean()

        rising = recent_avg > older_avg * 1.2  # 20% increase threshold

        return weekly_counts, recent_avg - older_avg, rising

    # --- Streamlit UI ---
    st.header("Upload a document to extract risk/event mentions (PDF, DOCX, or TXT)")
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])

    if uploaded_file:
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        elif file_type == "text/plain":
            text = extract_text_from_txt(uploaded_file)
        else:
            st.error("Unsupported file type.")
            text = ""

        if text:
            st.header("🔎 Risks and Events Found in Document")
            risk_event_matches = extract_semantic_risk_sentences(text, risk_event_definitions)

            if risk_event_matches:
                # --- Summary table without description ---
                st.subheader("📊 Summary of Detected Risks/Events")
                summary_data = [
                    {
                        "Category": category,
                        "Mentions": len(mentions)
                    }
                    for category, mentions in risk_event_matches.items()
                ]
                summary_df = pd.DataFrame(summary_data).sort_values(by="Mentions", ascending=False)
                st.dataframe(summary_df, use_container_width=True)

                st.markdown("---")

                # --- Detailed mentions ---
                st.subheader("📝 Detailed Mentions by Category")
                for category, mentions in risk_event_matches.items():
                    st.markdown(f"### ✅ {category}")
                    st.write(f"*{risk_event_definitions[category]['description']}*")
                    st.markdown("**Top Matches:**")
                    for sent, score in sorted(mentions, key=lambda x: -x[1])[:5]:
                        st.markdown(f"- `{score}`: {sent}")
                    st.markdown("---")

                # --- Emerging risk trends ---
                st.subheader("📈 Emerging Risk Trends")
                missing_trend_data = []
                for category in risk_event_matches.keys():
                    weekly_counts, diff, rising = check_risk_trend(category)
                    if weekly_counts is None:
                        missing_trend_data.append(category)
                        continue

                    chart = alt.Chart(weekly_counts).mark_line(point=True).encode(
                        x='Published:T',
                        y='mentions:Q',
                        tooltip=['Published:T', 'mentions:Q']
                    ).properties(width=500, height=250)

                    st.markdown(f"**{category}**")
                    st.altair_chart(chart, use_container_width=True)

                    if rising:
                        st.warning(f"⚠️ {category} has been on the rise in recent weeks. Consider allocating resources.")
                    else:
                        st.success(f"✅ {category} trend appears stable or declining.")

                if missing_trend_data:
                    st.info(
                        "No emerging trend data found for the following categories:\n" +
                        ", ".join([f"**{cat}**" for cat in missing_trend_data])
                    )

            else:
                st.info("No risk-related sentences matched semantically.")
        else:
            st.warning("No extractable text found in the document.")
