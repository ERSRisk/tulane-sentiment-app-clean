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
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
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
selection = st.sidebar.selectbox("Choose a tool:", ["External Risk Snapshot", "Article Risk Review", "Unmatched Topic Analysis", "Risk/Event Detector"])

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
    LIFECYCLE_LOCAL = (
        "pipeline/resources/"
        "risk_lifecycle_registry.csv"
    )

    # =========================================================
    # Helper functions
    # =========================================================

    @st.cache_data(
        show_spinner=True,
        ttl=1800,
    )

    def exclude_no_risk(data):
        data = data.copy()
    
        if "Predicted_Risks_new" in data.columns:
            risk_values = (
                data["Predicted_Risks_new"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.casefold()
            )
    
            data = data[
                risk_values.ne("no risk")
                & risk_values.ne("")
                & risk_values.ne("nan")
                & risk_values.ne("none")
            ].copy()
    
        if "Dashboard_Risk" in data.columns:
            dashboard_values = (
                data["Dashboard_Risk"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.casefold()
            )
    
            data = data[
                dashboard_values.ne("no risk")
                & dashboard_values.ne("")
                & dashboard_values.ne("nan")
                & dashboard_values.ne("none")
            ].copy()
    
        return data
    def get_csv_from_release(
        owner,
        repo,
        tag,
        asset,
        usecols=None,
    ) -> pd.DataFrame:
        token = _github_token()

        if not token:
            raise RuntimeError(
                "GITHUB_TOKEN missing "
                "(not injected or empty)."
            )

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"token {token}",
        }

        rel = requests.get(
            (
                "https://api.github.com/repos/"
                f"{owner}/{repo}/releases/tags/{tag}"
            ),
            headers=headers,
            timeout=60,
        )

        if rel.status_code != 200:
            raise RuntimeError(
                "Release lookup "
                f"{rel.status_code}: "
                f"{rel.text[:300]}"
            )

        rel_json = rel.json()

        asset_obj = next(
            (
                release_asset
                for release_asset
                in rel_json.get("assets", [])
                if release_asset.get("name") == asset
            ),
            None,
        )

        if not asset_obj:
            raise RuntimeError(
                f"Asset '{asset}' not found "
                f"in release '{tag}'."
            )

        url = asset_obj["browser_download_url"]

        response = requests.get(
            url,
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/octet-stream",
            },
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(
                "Asset download "
                f"{response.status_code}: "
                f"{response.text[:300]}"
            )

        return pd.read_csv(
            io.BytesIO(response.content),
            compression="gzip",
            low_memory=False,
            dtype=str,
            usecols=usecols,
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
                "The selected event was not found "
                "in the lifecycle registry."
            )

        now_utc = pd.Timestamp.now(tz="UTC")

        updated.loc[
            mask,
            "is_pinned",
        ] = bool(should_pin)

        updated.loc[
            mask,
            "pinned_at",
        ] = (
            now_utc
            if should_pin
            else pd.NaT
        )

        updated.loc[
            mask,
            "pinned_by",
        ] = (
            pinned_by
            if should_pin
            else ""
        )

        if should_pin:
            updated.loc[
                mask,
                "lifecycle_status",
            ] = "active"

            updated.loc[
                mask,
                "expired_at",
            ] = pd.NaT

        return updated

    def apply_risk_mapping(
        data,
        mapping,
    ):
        data = data.copy()
        mapping = mapping.copy()

        mapping["old_risk"] = (
            mapping["old_risk"]
            .astype(str)
            .str.strip()
        )

        mapping["dashboard_risk"] = (
            mapping["dashboard_risk"]
            .astype(str)
            .str.strip()
        )

        data["Predicted_Risks_new"] = (
            data["Predicted_Risks_new"]
            .fillna("No Risk")
            .astype(str)
            .str.strip()
        )

        data = data.merge(
            mapping,
            left_on="Predicted_Risks_new",
            right_on="old_risk",
            how="left",
        )

        data["Dashboard_Risk"] = (
            data["dashboard_risk"]
            .fillna(
                data["Predicted_Risks_new"]
            )
        )

        return data.drop(
            columns=[
                "old_risk",
                "dashboard_risk",
            ],
            errors="ignore",
        )

    def parse_article_sources(value):
        if isinstance(value, list):
            return value

        if pd.isna(value):
            return []

        text = str(value).strip()

        if not text:
            return []

        try:
            parsed = json.loads(text)

            return (
                parsed
                if isinstance(parsed, list)
                else []
            )

        except Exception:
            pass

        try:
            parsed = ast.literal_eval(text)

            return (
                parsed
                if isinstance(parsed, list)
                else []
            )

        except Exception:
            return []

    def parse_json_list(value):
        if isinstance(value, list):
            return value

        if pd.isna(value):
            return []

        try:
            parsed = json.loads(
                str(value)
            )

            return (
                parsed
                if isinstance(parsed, list)
                else []
            )

        except Exception:
            return []

    def prepare_agent_decisions(data):
        data = data.copy()

        if data.empty:
            return data

        if "evaluation_timestamp" in data.columns:
            data["evaluation_timestamp"] = (
                pd.to_datetime(
                    data["evaluation_timestamp"],
                    errors="coerce",
                    utc=True,
                )
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

        direct_mask = (
            data["institutional_relevance"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("direct")
        )

        visible_mask = (
            data["dashboard_visibility"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("show")
        )

        review_values = (
            data["requires_human_review"]
            .fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
        )

        no_review_mask = ~review_values.isin(
            [
                "true",
                "1",
                "yes",
            ]
        )

        data = data[
            direct_mask
            & visible_mask
            & no_review_mask
        ].copy()

        dedup_key = (
            "canonical_event_id"
            if "canonical_event_id"
            in data.columns
            else "unit_id"
        )

        if "evaluation_timestamp" in data.columns:
            data = data.sort_values(
                "evaluation_timestamp",
                ascending=False,
            )

        data = data.drop_duplicates(
            subset=[dedup_key],
            keep="first",
        )

        return data

    def executive_priority(row):
        decision = str(
            row.get(
                "agent_decision",
                "",
            )
        ).lower()

        actionability = str(
            row.get(
                "actionability",
                "",
            )
        ).lower()

        relevance = str(
            row.get(
                "institutional_relevance",
                "",
            )
        ).lower()

        if (
            decision == "escalate"
            and actionability == "high"
            and relevance == "direct"
        ):
            return "Critical"

        if decision == "escalate":
            return "High"

        if (
            decision == "monitor"
            and relevance == "direct"
        ):
            return "Medium"

        return "Low"

    def severity_bucket(x: float) -> str:
        if pd.isna(x):
            return "-"

        if x >= 4.0:
            return "Critical"

        if x >= 3.0:
            return "Elevated"

        if x >= 2.0:
            return "Monitor"

        return "Low"

    def action_label(
        score,
        trend,
        event_count,
    ):
        event_count = (
            0
            if pd.isna(event_count)
            else event_count
        )

        if (
            score >= 4
            or (
                score >= 3
                and trend > 0.3
            )
        ):
            return "Escalate"

        if (
            score >= 2
            or trend > 0.2
            or event_count >= 3
        ):
            return "Monitor"

        return "Watch"

    def clean_event_label(
        label,
        fallback_title,
    ):
        label = (
            ""
            if pd.isna(label)
            else str(label).strip()
        )

        if label.lower() in [
            "",
            "nan",
            "none",
        ]:
            if pd.notna(fallback_title):
                return str(
                    fallback_title
                )[:140]

            return "Unlabeled signal"

        return label

    def format_date(value):
        value = pd.to_datetime(
            value,
            errors="coerce",
            utc=True,
        )

        if pd.isna(value):
            return "Not available"

        return value.strftime(
            "%B %d, %Y"
        )

    @st.cache_data(
        show_spinner=False,
        ttl=3600,
    )
    def llm_relevance_filter(
        selected_risk,
        rows_json,
    ):
        api_key = os.getenv(
            "API_KEY_PAID"
        )

        if not api_key:
            return []

        client = genai.Client(
            api_key=api_key
        )

        prompt = f"""
You are filtering external news signals for a university
enterprise risk dashboard.

Selected dashboard risk:
{selected_risk}

Keep an item only if it is clearly relevant to Tulane
University in its operations or if it geographically affects
New Orleans or the Southeast United States.

Keep an item if it affects or could potentially affect higher
education institutions across the United States or institutions
similar in size to Tulane University.

Reject items that are merely general politics, unrelated sports,
general healthcare, celebrity news, or weakly connected noise.

Return ONLY valid JSON:

[
  {{
    "row_id": "...",
    "keep": true,
    "reason": "short reason"
  }}
]

Items:
{rows_json}
"""

        response = (
            client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
        )

        text = (
            response.text
            .strip()
            .replace(
                "```json",
                "",
            )
            .replace(
                "```",
                "",
            )
            .strip()
        )

        try:
            return json.loads(text)

        except Exception:
            return []

    # =========================================================
    # Load data
    # =========================================================

    articles = load_csv_gz_from_gcs(
        (
            "latest/topics/"
            "BERTopic_Streamlit.csv.gz"
        ),
        (
            "pipeline/resources/"
            "BERTopic_Streamlit.csv.gz"
        ),
    )

    df = load_csv_from_gcs(
        (
            "latest/dashboard/"
            "final_risk_scores1.csv"
        ),
        (
            "pipeline/resources/"
            "final_risk_scores1.csv"
        ),
    )

    events = load_csv_from_gcs(
        (
            "latest/dashboard/"
            "ranked_events_risks1.csv"
        ),
        (
            "pipeline/resources/"
            "ranked_events_risks1.csv"
        ),
    )

    agent_decisions = load_csv_from_gcs(
        "agent/agent_decisions.csv",
        (
            "pipeline/resources/"
            "agent_decisions.csv"
        ),
    )

    risk_lifecycle = load_csv_from_gcs(
        (
            "agent/"
            "risk_lifecycle_registry.csv"
        ),
        (
            "pipeline/resources/"
            "risk_lifecycle_registry.csv"
        ),
    )

    consolidated_packets = load_csv_from_gcs(
        (
            "agent/"
            "consolidated_agent_packets.csv"
        ),
        (
            "pipeline/resources/"
            "consolidated_agent_packets.csv"
        ),
    )

    emerging_situations = load_csv_from_gcs(
        (
            "agent/"
            "emerging_situations.csv"
        ),
        (
            "pipeline/resources/"
            "emerging_situations.csv"
        ),
    )

    risk_mapping = load_csv_from_gcs(
        (
            "latest/reference/"
            "risk_mapping.csv"
        ),
        (
            "pipeline/resources/"
            "risk_mapping.csv"
        ),
    )

    risk_mapping["old_risk"] = (
        risk_mapping["old_risk"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    
    risk_mapping["dashboard_risk"] = (
        risk_mapping["dashboard_risk"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    
    risk_mapping = risk_mapping[
        risk_mapping["old_risk"]
        .str.casefold()
        .ne("no risk")
        &
        risk_mapping["dashboard_risk"]
        .str.casefold()
        .ne("no risk")
    ].copy()

    # =========================================================
    # Prepare agent decisions
    # =========================================================

    for col in [
        "first_seen",
        "last_seen",
        "evaluation_timestamp",
    ]:
        if col in agent_decisions.columns:
            agent_decisions[col] = (
                pd.to_datetime(
                    agent_decisions[col],
                    errors="coerce",
                    utc=True,
                )
            )

    agent_decisions = (
        prepare_agent_decisions(
            agent_decisions
        )
    )

    if not agent_decisions.empty:
        agent_decisions[
            "Executive Priority"
        ] = agent_decisions.apply(
            executive_priority,
            axis=1,
        )

    # =========================================================
    # Prepare lifecycle registry
    # =========================================================

    LIFECYCLE_DATE_COLUMNS = [
        "first_seen",
        "last_seen",
        "last_evidence_at",
        "first_promoted_at",
        "last_qualified_at",
        "pinned_at",
        "expired_at",
    ]

    for column in LIFECYCLE_DATE_COLUMNS:
        if column in risk_lifecycle.columns:
            risk_lifecycle[column] = (
                pd.to_datetime(
                    risk_lifecycle[column],
                    errors="coerce",
                    utc=True,
                )
            )

    risk_lifecycle["is_pinned"] = (
        risk_lifecycle["is_pinned"]
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )

    # =========================================================
    # Build emerging risk dataset
    # =========================================================

    packet_source_columns = [
        "canonical_event_id",
        "top_articles",
        "top_titles",
        "source_packet_count",
    ]

    available_packet_columns = [
        column
        for column in packet_source_columns
        if column
        in consolidated_packets.columns
    ]

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
    ]

    available_decision_columns = [
        column
        for column in decision_columns
        if column
        in agent_decisions.columns
    ]

    emerging_risk_items = (
        risk_lifecycle
        .merge(
            agent_decisions[
                available_decision_columns
            ],
            on="canonical_event_id",
            how="left",
        )
        .merge(
            consolidated_packets[
                available_packet_columns
            ],
            on="canonical_event_id",
            how="left",
        )
    )

    if "top_risk_match" in emerging_risk_items.columns:
        emerging_risk_items = emerging_risk_items[
            emerging_risk_items[
                "top_risk_match"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
            .ne("no risk")
        ].copy()

    if (
        "top_articles"
        not in emerging_risk_items.columns
    ):
        emerging_risk_items[
            "top_articles"
        ] = None

    emerging_risk_items[
        "article_sources"
    ] = (
        emerging_risk_items[
            "top_articles"
        ]
        .apply(
            parse_article_sources
        )
    )

    emerging_risk_items = (
        emerging_risk_items[
            emerging_risk_items[
                "is_pinned"
            ]
            | emerging_risk_items[
                "lifecycle_status"
            ].isin(
                [
                    "new",
                    "candidate",
                    "active",
                    "cooling",
                ]
            )
        ]
        .copy()
    )

    LIFECYCLE_ORDER = {
        "new": 1,
        "active": 2,
        "candidate": 3,
        "cooling": 4,
        "expired": 5,
    }

    emerging_risk_items[
        "_lifecycle_order"
    ] = (
        emerging_risk_items[
            "lifecycle_status"
        ]
        .map(
            LIFECYCLE_ORDER
        )
        .fillna(99)
    )

    emerging_risk_items = (
        emerging_risk_items
        .sort_values(
            [
                "is_pinned",
                "_lifecycle_order",
                "priority_score",
                "last_evidence_at",
            ],
            ascending=[
                False,
                True,
                False,
                False,
            ],
            na_position="last",
        )
    )

    # =========================================================
    # Prepare risk mapping
    # =========================================================

    risk_universe = (
        risk_mapping[
            ["dashboard_risk"]
        ]
        .dropna()
        .copy()
    )

    risk_universe[
        "dashboard_risk"
    ] = (
        risk_universe[
            "dashboard_risk"
        ]
        .astype(str)
        .str.strip()
    )

    risk_universe = (
        risk_universe[
            risk_universe[
                "dashboard_risk"
            ].ne("")
        ]
        .drop_duplicates()
        .rename(
            columns={
                "dashboard_risk":
                "Dashboard_Risk"
            }
        )
    )

    df = apply_risk_mapping(
        df,
        risk_mapping,
    )

    events = apply_risk_mapping(
        events,
        risk_mapping,
    )

    articles = apply_risk_mapping(
        articles,
        risk_mapping,
    )

    df = exclude_no_risk(df)
    events = exclude_no_risk(events)
    articles = exclude_no_risk(articles)



    # =========================================================
    # Calculate snapshot data
    # =========================================================

    period_map = {
        "Last Month": 30,
        "Last 3 Months": 90,
        "Last 6 Months": 180,
        "Last Year": 365,
    }

    selected_period = "Last Month"

    delta_days = period_map[
        selected_period
    ]

    delta = pd.Timedelta(
        days=delta_days
    )

    events["Window"] = pd.to_datetime(
        events["Window"],
        errors="coerce",
        utc=True,
    )

    df["Window"] = pd.to_datetime(
        df["Window"],
        errors="coerce",
        utc=True,
    )

    cutoff = (
        pd.Timestamp.now(tz="UTC")
        - delta
    )

    previous_cutoff = (
        cutoff
        - delta
    )

    current_events = events[
        events["Window"] >= cutoff
    ].copy()

    previous_events = events[
        (
            events["Window"]
            >= previous_cutoff
        )
        & (
            events["Window"]
            < cutoff
        )
    ].copy()

    total_events = (
        current_events.shape[0]
    )

    events_summary = (
        current_events
        .groupby(
            [
                "Window",
                "Dashboard_Risk",
            ]
        )["Title"]
        .count()
        .reset_index()
        .rename(
            columns={
                "Title":
                "Event_Count"
            }
        )
    )

    events[
        "Event_Severity"
    ] = pd.to_numeric(
        events[
            "Event_Severity"
        ],
        errors="coerce",
    ).fillna(0)

    current_events[
        "Event_Severity"
    ] = pd.to_numeric(
        current_events[
            "Event_Severity"
        ],
        errors="coerce",
    ).fillna(0)

    previous_events[
        "Event_Severity"
    ] = pd.to_numeric(
        previous_events[
            "Event_Severity"
        ],
        errors="coerce",
    ).fillna(0)

    current_scores = (
        current_events
        .groupby(
            "Dashboard_Risk"
        )[
            "Event_Severity"
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "Event_Severity":
                "current_score"
            }
        )
    )

    previous_scores = (
        previous_events
        .groupby(
            "Dashboard_Risk"
        )[
            "Event_Severity"
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "Event_Severity":
                "previous_score"
            }
        )
    )

    snapshot = (
        risk_universe
        .merge(
            current_scores,
            on="Dashboard_Risk",
            how="left",
        )
        .merge(
            previous_scores,
            on="Dashboard_Risk",
            how="left",
        )
    )

    snapshot[
        "current_score"
    ] = pd.to_numeric(
        snapshot[
            "current_score"
        ],
        errors="coerce",
    ).fillna(0.0)

    snapshot[
        "previous_score"
    ] = pd.to_numeric(
        snapshot[
            "previous_score"
        ],
        errors="coerce",
    ).fillna(0.0)

    current_risk_names = set(
        current_scores[
            "Dashboard_Risk"
        ]
        .dropna()
        .astype(str)
    )

    previous_risk_names = set(
        previous_scores[
            "Dashboard_Risk"
        ]
        .dropna()
        .astype(str)
    )

    snapshot["is_new"] = (
        snapshot[
            "Dashboard_Risk"
        ].isin(
            current_risk_names
        )
        & ~snapshot[
            "Dashboard_Risk"
        ].isin(
            previous_risk_names
        )
    )

    snapshot["trend"] = (
        snapshot["current_score"]
        - snapshot["previous_score"]
    )

    snapshot.loc[
        snapshot["is_new"],
        "trend",
    ] = snapshot.loc[
        snapshot["is_new"],
        "current_score",
    ]

    snapshot[
        "trend_display"
    ] = snapshot[
        "trend"
    ].apply(
        lambda value: (
            "🔺"
            if value > 0.2
            else (
                "🔻"
                if value < -0.2
                else " "
            )
        )
    )

    snapshot[
        "severity_band"
    ] = snapshot[
        "current_score"
    ].apply(
        severity_bucket
    )

    event_counts = (
        current_events
        .groupby(
            "Dashboard_Risk"
        )
        .size()
        .reset_index(
            name="Event_Count"
        )
    )

    snapshot = snapshot.merge(
        event_counts,
        on="Dashboard_Risk",
        how="left",
    )

    snapshot[
        "Event_Count"
    ] = (
        snapshot[
            "Event_Count"
        ]
        .fillna(0)
        .astype(int)
    )

    snapshot["action"] = (
        snapshot.apply(
            lambda row: action_label(
                row[
                    "current_score"
                ],
                row["trend"],
                row[
                    "Event_Count"
                ],
            ),
            axis=1,
        )
    )

    total_score = (
        snapshot[
            "current_score"
        ].sum()
    )

    if total_score > 0:
        snapshot[
            "risk_share_pct"
        ] = (
            snapshot[
                "current_score"
            ]
            / total_score
        ) * 100

    else:
        snapshot[
            "risk_share_pct"
        ] = 0.0

    high_risk_count = snapshot[
        snapshot[
            "current_score"
        ] > 3.0
    ].shape[0]

    emerging_count = snapshot[
        (
            snapshot["trend"] > 0.2
        )
        & (
            snapshot[
                "current_score"
            ] < 3
        )
    ].shape[0]

    persistent_count = snapshot[
        (
            snapshot[
                "trend"
            ].abs() <= 1.0
        )
        & (
            snapshot[
                "current_score"
            ] > 3.0
        )
    ].shape[0]

    persistent_risks = snapshot[
        (
            snapshot[
                "trend"
            ].abs() <= 1.0
        )
        & (
            snapshot[
                "current_score"
            ] > 3.0
        )
    ][
        "Dashboard_Risk"
    ]

    persistent_risks_names = (
        persistent_risks.tolist()
        if len(persistent_risks)
        else "-"
    )

    top_trending = (
        snapshot
        .sort_values(
            "trend",
            ascending=False,
        )
        .head(5)
    )

    top_high = (
        snapshot
        .sort_values(
            "current_score",
            ascending=False,
        )
        .head(8)
    )

    new_risks = (
        snapshot[
            snapshot["is_new"]
        ]
        .sort_values(
            "current_score",
            ascending=False,
        )
        .head(5)
    )

    df = df.merge(
        snapshot[
            [
                "Dashboard_Risk",
                "trend_display",
                "risk_share_pct",
                "current_score",
            ]
        ],
        on="Dashboard_Risk",
        how="left",
    )

    df = df.merge(
        events_summary[
            [
                "Dashboard_Risk",
                "Window",
                "Event_Count",
            ]
        ],
        on=[
            "Dashboard_Risk",
            "Window",
        ],
        how="left",
    )

    top_risk = (
        snapshot
        .sort_values(
            "current_score",
            ascending=False,
        )
        .head(1)
    )

    top_risk_name = (
        top_risk[
            "Dashboard_Risk"
        ].iloc[0]
        if len(top_risk)
        else "-"
    )

    top_risk_score = (
        top_risk[
            "current_score"
        ].iloc[0]
        if len(top_risk)
        else 0
    )

    fastest = (
        snapshot
        .sort_values(
            "trend",
            ascending=False,
        )
        .head(1)
    )

    fastest_name = (
        fastest[
            "Dashboard_Risk"
        ].iloc[0]
        if len(fastest)
        else "-"
    )

    fastest_delta = (
        fastest[
            "trend"
        ].iloc[0]
        if len(fastest)
        else 0
    )

    # =========================================================
    # Page tabs
    # =========================================================

    st.title(
        "External Risk Intelligence"
    )

    snapshot_tab, risk_details_tab = st.tabs(
        [
            "External Risk Snapshot",
            "Risk Overview & Stories",
        ]
    )

    # =========================================================
    # TAB 1: Snapshot and emerging developments
    # =========================================================

    with snapshot_tab:
        st.subheader(
            "External Risk Snapshot"
        )

        st.caption(
            "Executive view of external risk activity "
            f"for the {selected_period.lower()}."
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Total Events for Period",
            total_events,
        )

        col2.metric(
            "Highest Risk",
            f"{top_risk_score:.2f}",
            help=top_risk_name,
        )

        col3.metric(
            "Fastest Growth",
            f"{fastest_delta:.2f}",
            help=fastest_name,
        )

        col4.metric(
            "Persistent High Risks",
            persistent_count,
        )

        st.divider()

        st.subheader(
            "Trending Risks"
        )

        nonzero_trending = (
            top_trending[
                top_trending[
                    "trend"
                ].ne(0)
            ]
            .copy()
        )

        if nonzero_trending.empty:
            st.caption(
                "No material risk trend changes "
                "were detected for this period."
            )

        else:
            trend_cols = st.columns(
                len(nonzero_trending)
            )

            for column, (_, row) in zip(
                trend_cols,
                nonzero_trending.iterrows(),
            ):
                with column:
                    st.metric(
                        label=row[
                            "Dashboard_Risk"
                        ],
                        value=(
                            f"{row['current_score']:.2f}"
                        ),
                        delta=(
                            f"{row['trend']:.2f}"
                        ),
                        delta_color="inverse",
                    )

        st.divider()

        st.subheader(
            "Emerging Risk Developments"
        )

        st.caption(
            "Risks remain visible while evidence is current. "
            "Items cool and eventually leave the main view "
            "when no new evidence appears. "
            "Pinned items remain active."
        )

        status_filter = st.multiselect(
            "Lifecycle Status",
            [
                "new",
                "active",
                "candidate",
                "cooling",
            ],
            default=[
                "new",
                "active",
                "candidate",
                "cooling",
            ],
            key=(
                "external_risk_"
                "lifecycle_status"
            ),
        )

        display_items = (
            emerging_risk_items[
                emerging_risk_items[
                    "lifecycle_status"
                ].isin(
                    status_filter
                )
            ]
            .copy()
        )

        pinned_count = int(
            display_items[
                "is_pinned"
            ].sum()
        )

        new_count = int(
            display_items[
                "lifecycle_status"
            ]
            .eq("new")
            .sum()
        )

        active_count = int(
            display_items[
                "lifecycle_status"
            ]
            .eq("active")
            .sum()
        )

        cooling_count = int(
            display_items[
                "lifecycle_status"
            ]
            .eq("cooling")
            .sum()
        )

        metric_1, metric_2, metric_3, metric_4 = (
            st.columns(4)
        )

        metric_1.metric(
            "New Developments",
            new_count,
        )

        metric_2.metric(
            "Active Developments",
            active_count,
        )

        metric_3.metric(
            "Cooling",
            cooling_count,
        )

        metric_4.metric(
            "Pinned",
            pinned_count,
        )

        STATUS_LABELS = {
            "new": "New",
            "active": "Active",
            "candidate": "Candidate",
            "cooling": "Cooling",
            "expired": "Expired",
        }

        sort_choice = st.selectbox(
            "Sort emerging risks by",
            [
                "Newest evidence first",
                "Priority: highest first",
                "Oldest evidence first",
                "Recently promoted",
                "Lifecycle status",
            ],
            key=(
                "external_risk_"
                "emerging_sort"
            ),
        )

        if (
            sort_choice
            == "Priority: highest first"
        ):
            display_items = (
                display_items
                .sort_values(
                    [
                        "is_pinned",
                        "priority_score",
                        "last_evidence_at",
                    ],
                    ascending=[
                        False,
                        False,
                        False,
                    ],
                    na_position="last",
                )
            )

        elif (
            sort_choice
            == "Newest evidence first"
        ):
            display_items = (
                display_items
                .sort_values(
                    [
                        "is_pinned",
                        "last_seen",
                        "priority_score",
                    ],
                    ascending=[
                        False,
                        False,
                        False,
                    ],
                    na_position="last",
                )
            )

        elif (
            sort_choice
            == "Oldest evidence first"
        ):
            display_items = (
                display_items
                .sort_values(
                    [
                        "is_pinned",
                        "last_seen",
                        "priority_score",
                    ],
                    ascending=[
                        False,
                        True,
                        False,
                    ],
                    na_position="last",
                )
            )

        elif (
            sort_choice
            == "Recently promoted"
        ):
            display_items = (
                display_items
                .sort_values(
                    [
                        "is_pinned",
                        "first_promoted_at",
                        "priority_score",
                    ],
                    ascending=[
                        False,
                        False,
                        False,
                    ],
                    na_position="last",
                )
            )

        elif (
            sort_choice
            == "Lifecycle status"
        ):
            display_items[
                "_status_order"
            ] = (
                display_items[
                    "lifecycle_status"
                ]
                .map(
                    {
                        "new": 1,
                        "active": 2,
                        "candidate": 3,
                        "cooling": 4,
                        "expired": 5,
                    }
                )
                .fillna(99)
            )

            display_items = (
                display_items
                .sort_values(
                    [
                        "is_pinned",
                        "_status_order",
                        "priority_score",
                    ],
                    ascending=[
                        False,
                        True,
                        False,
                    ],
                    na_position="last",
                )
            )

        if display_items.empty:
            st.info(
                "No emerging risk developments "
                "match the selected lifecycle statuses."
            )

        for _, row in display_items.iterrows():
            canonical_id = str(
                row[
                    "canonical_event_id"
                ]
            )

            raw_title = row.get(
                "situation_title",
                "Untitled risk development",
            )

            situation_title = (
                str(raw_title).strip()
                if pd.notna(raw_title)
                else "Untitled risk development"
            )

            if situation_title.lower() in [
                "",
                "nan",
                "none",
            ]:
                situation_title = (
                    "Untitled risk development"
                )

            lifecycle_status = str(
                row.get(
                    "lifecycle_status",
                    "candidate",
                )
            ).lower()

            is_pinned = bool(
                row.get(
                    "is_pinned",
                    False,
                )
            )

            raw_risk_name = row.get(
                "top_risk_match",
                "Unmapped risk",
            )

            risk_name = (
                str(raw_risk_name).strip()
                if pd.notna(raw_risk_name)
                else "Unmapped risk"
            )

            if risk_name.lower() in [
                "",
                "nan",
                "none",
            ]:
                risk_name = "Unmapped risk"

            status_label = (
                STATUS_LABELS.get(
                    lifecycle_status,
                    lifecycle_status.title(),
                )
            )

            pin_label = (
                "Pinned"
                if is_pinned
                else status_label
            )

            with st.container(
                border=True
            ):
                title_col, status_col = (
                    st.columns(
                        [4, 1]
                    )
                )

                with title_col:
                    st.markdown(
                        f"#### {situation_title}"
                    )

                    st.caption(
                        f"{risk_name} · {pin_label}"
                    )

                with status_col:
                    if is_pinned:
                        st.markdown(
                            "**Pinned indefinitely**"
                        )

                    else:
                        st.markdown(
                            f"**{status_label}**"
                        )

                (
                    detail_col_1,
                    detail_col_2,
                    detail_col_3,
                ) = st.columns(3)

                priority_score = pd.to_numeric(
                    row.get(
                        "priority_score",
                        0,
                    ),
                    errors="coerce",
                )

                if pd.isna(priority_score):
                    priority_score = 0

                detail_col_1.metric(
                    "Priority Score",
                    f"{priority_score:.2f}",
                )

                detail_col_2.metric(
                    "First Seen",
                    format_date(
                        row.get(
                            "first_seen"
                        )
                    ),
                )

                detail_col_3.metric(
                    "Latest Evidence",
                    format_date(
                        row.get(
                            "last_evidence_at"
                        )
                    ),
                )

                what_changed = row.get(
                    "what_changed"
                )

                if (
                    pd.notna(what_changed)
                    and str(
                        what_changed
                    ).strip()
                ):
                    st.markdown(
                        "**What changed**"
                    )

                    st.write(
                        what_changed
                    )

                why_it_matters = row.get(
                    "why_it_matters"
                )

                if (
                    pd.notna(
                        why_it_matters
                    )
                    and str(
                        why_it_matters
                    ).strip()
                ):
                    st.markdown(
                        "**Why it matters to Tulane**"
                    )

                    st.write(
                        why_it_matters
                    )

                recommended_action = row.get(
                    "recommended_human_action"
                )

                if (
                    pd.notna(
                        recommended_action
                    )
                    and str(
                        recommended_action
                    ).strip()
                ):
                    st.markdown(
                        "**Recommended action**"
                    )

                    st.write(
                        recommended_action
                    )

                article_sources = row.get(
                    "article_sources",
                    [],
                )

                if not isinstance(
                    article_sources,
                    list,
                ):
                    article_sources = (
                        parse_article_sources(
                            article_sources
                        )
                    )

                if article_sources:
                    with st.expander(
                        (
                            "Read supporting sources "
                            f"({len(article_sources)})"
                        )
                    ):
                        for (
                            article_number,
                            article,
                        ) in enumerate(
                            article_sources,
                            start=1,
                        ):
                            if not isinstance(
                                article,
                                dict,
                            ):
                                continue

                            article_title = str(
                                article.get(
                                    "title",
                                    (
                                        "Source "
                                        f"{article_number}"
                                    ),
                                )
                            ).strip()

                            article_link = str(
                                article.get(
                                    "link",
                                    "",
                                )
                            ).strip()

                            article_date = (
                                pd.to_datetime(
                                    article.get(
                                        "published"
                                    ),
                                    errors="coerce",
                                    utc=True,
                                )
                            )

                            article_snippet = str(
                                article.get(
                                    "snippet",
                                    "",
                                )
                            ).strip()

                            if (
                                article_link
                                .startswith("http")
                            ):
                                st.markdown(
                                    (
                                        f"**[{article_title}]"
                                        f"({article_link})**"
                                    )
                                )

                            else:
                                st.markdown(
                                    f"**{article_title}**"
                                )

                            if pd.notna(
                                article_date
                            ):
                                st.caption(
                                    article_date
                                    .strftime(
                                        "%B %d, %Y"
                                    )
                                )

                            if article_snippet:
                                st.write(
                                    article_snippet
                                )

                            if (
                                article_number
                                < len(
                                    article_sources
                                )
                            ):
                                st.divider()

                pin_col, lifecycle_col = (
                    st.columns(
                        [1, 3]
                    )
                )

                with pin_col:
                    button_text = (
                        "Unpin"
                        if is_pinned
                        else "Keep indefinitely"
                    )

                    if st.button(
                        button_text,
                        key=(
                            "risk_pin_"
                            f"{canonical_id}"
                        ),
                        use_container_width=True,
                    ):
                        try:
                            risk_lifecycle = (
                                update_risk_pin(
                                    lifecycle_df=(
                                        risk_lifecycle
                                    ),
                                    canonical_event_id=(
                                        canonical_id
                                    ),
                                    should_pin=(
                                        not is_pinned
                                    ),
                                )
                            )

                            save_lifecycle_registry_to_gcs(
                                lifecycle_df=(
                                    risk_lifecycle
                                ),
                                blob_name=(
                                    LIFECYCLE_BLOB
                                ),
                                local_path=(
                                    LIFECYCLE_LOCAL
                                ),
                            )

                            st.cache_data.clear()

                            st.success(
                                (
                                    "Risk pinned "
                                    "indefinitely."
                                )
                                if not is_pinned
                                else "Risk unpinned."
                            )

                            st.rerun()

                        except Exception as error:
                            st.error(
                                "The lifecycle registry "
                                "could not be updated: "
                                f"{error}"
                            )

                with lifecycle_col:
                    if is_pinned:
                        st.caption(
                            "This development will "
                            "remain active until a user "
                            "removes the pin."
                        )

                    elif (
                        lifecycle_status
                        == "cooling"
                    ):
                        st.caption(
                            "No recent supporting evidence "
                            "was detected. This item will "
                            "expire if the condition continues."
                        )

                    elif (
                        lifecycle_status
                        == "candidate"
                    ):
                        st.caption(
                            "This is an early signal that "
                            "has not yet met the promotion "
                            "threshold."
                        )

                    else:
                        st.caption(
                            "The weekly lifecycle process "
                            "will reassess this item when "
                            "new evidence is available."
                        )

    # =========================================================
    # TAB 2: Risk overview and stories
    # =========================================================

    with risk_details_tab:
        st.subheader(
            "Risk Overview"
        )

        table = snapshot.copy()

        table = table[
            [
                "Dashboard_Risk",
                "action",
                "severity_band",
                "current_score",
                "previous_score",
                "trend",
                "trend_display",
                "risk_share_pct",
                "Event_Count",
            ]
        ]

        table = table.sort_values(
            "Dashboard_Risk",
            ascending=True,
        )

        table = table.rename(
            columns={
                "Dashboard_Risk":
                "Risk",
                "action":
                "Action",
                "severity_band":
                "Severity Band",
                "current_score":
                "Current Score",
                "previous_score":
                "Previous Score",
                "trend":
                "Change",
                "trend_display":
                "Trend",
                "risk_share_pct":
                "Risk Share %",
                "Event_Count":
                "Event Count",
            }
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Current Score":
                st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Previous Score":
                st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Change":
                st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Risk Share %":
                st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
            },
        )

        st.divider()

        max_items = st.slider(
            "Maximum items to show",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key=(
                "risk_overview_"
                "maximum_items"
            ),
        )

        left, right = st.columns(
            [1, 2]
        )

        with left:
            all_dashboard_risks = sorted(
                risk_mapping[
                    "dashboard_risk"
                ]
                .dropna()
                .astype(str)
                .str.strip()
                .loc[
                    lambda values:
                    values.ne("")
                ]
                .unique(),
                key=str.lower,
            )

            if not all_dashboard_risks:
                st.error(
                    "No dashboard risks were "
                    "found in the risk mapping."
                )

                st.stop()

            selected_risk = st.selectbox(
                "Main Risk Register Category",
                all_dashboard_risks,
                key=(
                    "risk_overview_"
                    "selected_risk"
                ),
            )

            subcategory_options = sorted(
                risk_mapping.loc[
                    risk_mapping[
                        "dashboard_risk"
                    ]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .eq(
                        selected_risk
                    ),
                    "old_risk",
                ]
                .dropna()
                .astype(str)
                .str.strip()
                .loc[
                    lambda values:
                    values.ne("")
                ]
                .unique(),
                key=str.lower,
            )

            selected_subcategory = (
                st.selectbox(
                    "AI Risk Subcategory",
                    [
                        "All Subcategories"
                    ]
                    + subcategory_options,
                    key=(
                        "risk_overview_"
                        "subcategory"
                    ),
                )
            )

            min_sev = st.slider(
                "Minimum event severity",
                min_value=0.0,
                max_value=5.0,
                value=2.5,
                step=0.1,
                key=(
                    "risk_overview_"
                    "minimum_severity"
                ),
            )

            search = st.text_input(
                "Search headlines",
                "",
                key=(
                    "risk_overview_"
                    "headline_search"
                ),
            )

        with right:
            rrow = snapshot[
                snapshot[
                    "Dashboard_Risk"
                ].eq(
                    selected_risk
                )
            ].head(1)

            if not rrow.empty:
                rrow = rrow.iloc[0]

                event_count = (
                    int(
                        rrow[
                            "Event_Count"
                        ]
                    )
                    if pd.notna(
                        rrow[
                            "Event_Count"
                        ]
                    )
                    else 0
                )

                if rrow["trend"] > 0:
                    trend_word = (
                        "increased"
                    )

                elif rrow["trend"] < 0:
                    trend_word = (
                        "decreased"
                    )

                else:
                    trend_word = (
                        "stayed flat"
                    )

                why_action = (
                    "Recommended action is "
                    f"**{rrow['action']}** "
                    "because the current risk "
                    "score is "
                    f"**{rrow['current_score']:.2f}**, "
                    f"the score {trend_word} by "
                    f"**{abs(rrow['trend']):.2f}** "
                    "versus the prior period, and "
                    f"**{event_count}** event(s) "
                    "appeared in the selected period."
                )

                st.info(
                    why_action
                )

                metric_a, metric_b = (
                    st.columns(2)
                )

                metric_a.metric(
                    "Current Severity",
                    (
                        f"{rrow['current_score']:.2f}"
                    ),
                    delta=(
                        f"{rrow['trend']:.2f}"
                    ),
                    delta_color="inverse",
                )

                metric_b.metric(
                    "Event Count",
                    event_count,
                )

                st.markdown(
                    (
                        f"**Risk category:** "
                        f"{selected_risk}  \n"
                        f"**Severity band:** "
                        f"{rrow['severity_band']}  \n"
                        f"**Share of risk signal:** "
                        f"{rrow['risk_share_pct']:.1f}%  \n"
                        f"**Recommended action:** "
                        f"{rrow['action']}"
                    )
                )

        st.divider()

        # =====================================================
        # Build selected risk evidence dataset
        # =====================================================

        events_for_details = (
            events.copy()
        )

        articles_for_details = (
            articles.copy()
        )

        events_for_details[
            "Window"
        ] = pd.to_datetime(
            events_for_details[
                "Window"
            ],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)

        cutoff_naive = (
            pd.Timestamp(
                cutoff
            )
            .tz_convert(None)
        )

        risk_events = (
            events_for_details[
                (
                    events_for_details[
                        "Dashboard_Risk"
                    ]
                    == selected_risk
                )
                & (
                    events_for_details[
                        "Window"
                    ]
                    >= cutoff_naive
                )
            ]
            .copy()
        )

        if (
            selected_subcategory
            != "All Subcategories"
        ):
            risk_events = (
                risk_events[
                    risk_events[
                        "Predicted_Risks_new"
                    ]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .eq(
                        selected_subcategory
                    )
                ]
                .copy()
            )

        risk_events[
            "Event_Severity"
        ] = pd.to_numeric(
            risk_events[
                "Event_Severity"
            ],
            errors="coerce",
        )

        risk_events = (
            risk_events[
                risk_events[
                    "Event_Severity"
                ]
                .fillna(0)
                .ge(
                    min_sev
                )
            ]
            .copy()
        )

        article_detail_cols = [
            "Event_Label",
            "Title",
            "Content",
            "Published_utc",
            "Link",
            "Source",
            "source",
            "canonical_source",
        ]

        article_detail_cols = [
            column
            for column in article_detail_cols
            if column
            in articles_for_details.columns
        ]

        for detail_data in [
            risk_events,
            articles_for_details,
        ]:
            if (
                "Link"
                in detail_data.columns
            ):
                detail_data[
                    "Link"
                ] = (
                    detail_data[
                        "Link"
                    ]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

            if (
                "Title"
                in detail_data.columns
            ):
                detail_data[
                    "Title"
                ] = (
                    detail_data[
                        "Title"
                    ]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

        has_event_links = (
            "Link"
            in risk_events.columns
            and risk_events[
                "Link"
            ]
            .astype(str)
            .str.startswith(
                "http"
            )
            .any()
        )

        has_article_links = (
            "Link"
            in articles_for_details.columns
        )

        if (
            has_event_links
            and has_article_links
        ):
            join_keys = [
                "Link"
            ]

        elif (
            "Title"
            in risk_events.columns
            and "Title"
            in articles_for_details.columns
        ):
            join_keys = [
                "Title"
            ]

        else:
            join_keys = [
                "Event_Label"
            ]

        available_join_keys = [
            key
            for key in join_keys
            if key in article_detail_cols
        ]

        if available_join_keys:
            article_details = (
                articles_for_details[
                    article_detail_cols
                ]
                .drop_duplicates(
                    subset=(
                        available_join_keys
                    )
                )
            )

            risk_events = risk_events.merge(
                article_details,
                on=available_join_keys,
                how="left",
                suffixes=(
                    "",
                    "_article",
                ),
            )

        for col in [
            "Title",
            "Content",
            "Published_utc",
            "Link",
            "Source",
            "source",
            "canonical_source",
        ]:
            article_col = (
                f"{col}_article"
            )

            if (
                article_col
                in risk_events.columns
            ):
                if col in risk_events.columns:
                    existing_value = (
                        risk_events[col]
                        .notna()
                        & risk_events[
                            col
                        ]
                        .astype(str)
                        .str.strip()
                        .ne("")
                    )

                    risk_events[
                        col
                    ] = (
                        risk_events[
                            col
                        ]
                        .where(
                            existing_value,
                            risk_events[
                                article_col
                            ],
                        )
                    )

                else:
                    risk_events[
                        col
                    ] = risk_events[
                        article_col
                    ]

        risk_events = (
            risk_events.drop(
                columns=[
                    column
                    for column
                    in risk_events.columns
                    if column.endswith(
                        "_article"
                    )
                ],
                errors="ignore",
            )
        )

        if (
            search.strip()
            and "Title"
            in risk_events.columns
        ):
            risk_events = (
                risk_events[
                    risk_events[
                        "Title"
                    ]
                    .astype(str)
                    .str.contains(
                        search,
                        case=False,
                        na=False,
                    )
                ]
                .copy()
            )

        risk_events = (
            risk_events
            .sort_values(
                "Event_Severity",
                ascending=False,
            )
        )

        # =====================================================
        # Most Relevant Recent Stories
        # =====================================================

        st.markdown(
            "### Most Relevant Recent Stories"
        )

        articles_for_risk = (
            articles_for_details.copy()
        )

        if (
            "Published_utc"
            in articles_for_risk.columns
        ):
            articles_for_risk[
                "Published_utc"
            ] = pd.to_datetime(
                articles_for_risk[
                    "Published_utc"
                ],
                errors="coerce",
                utc=True,
            ).dt.tz_convert(None)

        elif (
            "Published"
            in articles_for_risk.columns
        ):
            articles_for_risk[
                "Published_utc"
            ] = pd.to_datetime(
                articles_for_risk[
                    "Published"
                ],
                errors="coerce",
                utc=True,
            ).dt.tz_convert(None)

        else:
            articles_for_risk[
                "Published_utc"
            ] = pd.NaT

        articles_for_risk = (
            articles_for_risk[
                (
                    articles_for_risk[
                        "Dashboard_Risk"
                    ]
                    == selected_risk
                )
                & (
                    articles_for_risk[
                        "Published_utc"
                    ]
                    >= cutoff_naive
                )
            ]
            .copy()
        )

        if (
            selected_subcategory
            != "All Subcategories"
        ):
            articles_for_risk = (
                articles_for_risk[
                    articles_for_risk[
                        "Predicted_Risks_new"
                    ]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .eq(
                        selected_subcategory
                    )
                ]
                .copy()
            )

        if (
            "University Label"
            in articles_for_risk.columns
        ):
            articles_for_risk[
                "University Label"
            ] = pd.to_numeric(
                articles_for_risk[
                    "University Label"
                ],
                errors="coerce",
            ).fillna(0).astype(int)

            articles_for_risk = (
                articles_for_risk[
                    articles_for_risk[
                        "University Label"
                    ]
                    == 1
                ]
                .copy()
            )

        if (
            search.strip()
            and "Title"
            in articles_for_risk.columns
        ):
            articles_for_risk = (
                articles_for_risk[
                    articles_for_risk[
                        "Title"
                    ]
                    .astype(str)
                    .str.contains(
                        search,
                        case=False,
                        na=False,
                    )
                ]
                .copy()
            )

        story_dedup_columns = [
            column
            for column in [
                "Title",
                "Link",
            ]
            if column
            in articles_for_risk.columns
        ]

        articles_for_risk = (
            articles_for_risk
            .dropna(
                subset=["Title"]
            )
        )

        if story_dedup_columns:
            articles_for_risk = (
                articles_for_risk
                .drop_duplicates(
                    subset=(
                        story_dedup_columns
                    )
                )
            )

        articles_for_risk = (
            articles_for_risk
            .sort_values(
                "Published_utc",
                ascending=False,
            )
            .head(
                max_items
            )
        )

        if articles_for_risk.empty:
            st.caption(
                "No article-level stories "
                "were found for the selected risk."
            )

        else:
            for _, row in (
                articles_for_risk.iterrows()
            ):
                with st.container(
                    border=True
                ):
                    title = row.get(
                        "Title",
                        "Untitled article",
                    )

                    link = str(
                        row.get(
                            "Link",
                            "",
                        )
                    ).strip()

                    if link.startswith(
                        "http"
                    ):
                        st.markdown(
                            f"**[{title}]({link})**"
                        )

                    else:
                        st.markdown(
                            f"**{title}**"
                        )

                    meta = []

                    source = row.get(
                        "Source",
                        row.get(
                            "source",
                            row.get(
                                "canonical_source",
                                "",
                            ),
                        ),
                    )

                    if (
                        pd.notna(source)
                        and str(
                            source
                        ).strip()
                    ):
                        meta.append(
                            str(source)
                        )

                    published = row.get(
                        "Published_utc"
                    )

                    if pd.notna(
                        published
                    ):
                        published_text = (
                            pd.to_datetime(
                                published,
                                errors="coerce",
                            )
                        )

                        if pd.notna(
                            published_text
                        ):
                            meta.append(
                                (
                                    "Published: "
                                    f"{published_text:%B %d, %Y}"
                                )
                            )

                    if meta:
                        st.caption(
                            " | ".join(meta)
                        )

                    content = row.get(
                        "Content"
                    )

                    if (
                        pd.notna(content)
                        and str(
                            content
                        ).strip()
                    ):
                        content_text = str(
                            content
                        )

                        preview = (
                            content_text[:500]
                        )

                        st.write(
                            preview
                            + (
                                "..."
                                if len(
                                    content_text
                                ) > 500
                                else ""
                            )
                        )

        st.divider()

        # =====================================================
        # Key drivers
        # =====================================================

        st.markdown(
            "### Why This Risk Is Showing Up"
        )

        driver_cols = [
            "Acceleration_value",
            "Recency",
            "Source_Accuracy",
            "Impact_Score",
            "Location",
            "Industry_Risk",
            "Frequency_Score",
        ]

        driver_map = {
            "Acceleration_value":
            "Momentum",
            "Recency":
            "Recency",
            "Impact_Score":
            "Impact",
            "Industry_Risk":
            "Higher-Ed Relevance",
            "Location":
            "Location Relevance",
            "Frequency_Score":
            "Event Frequency",
            "Source_Accuracy":
            "Source Reliability",
        }

        available_driver_cols = [
            column
            for column in driver_cols
            if column
            in risk_events.columns
        ]

        if (
            available_driver_cols
            and not risk_events.empty
        ):
            for column in (
                available_driver_cols
            ):
                risk_events[
                    column
                ] = pd.to_numeric(
                    risk_events[
                        column
                    ],
                    errors="coerce",
                )

            driver_summary = (
                risk_events[
                    available_driver_cols
                ]
                .mean()
                .reset_index()
                .rename(
                    columns={
                        "index":
                        "technical_driver",
                        0:
                        "contribution",
                    }
                )
            )

            driver_summary[
                "driver"
            ] = (
                driver_summary[
                    "technical_driver"
                ]
                .map(
                    driver_map
                )
            )

            driver_summary[
                "contribution"
            ] = (
                driver_summary[
                    "contribution"
                ]
                .round(2)
            )

            driver_summary[
                "level"
            ] = driver_summary[
                "contribution"
            ].apply(
                lambda value: (
                    "High"
                    if value >= 4
                    else (
                        "Moderate"
                        if value >= 2.5
                        else "Low"
                    )
                )
            )

            driver_summary = (
                driver_summary[
                    [
                        "driver",
                        "level",
                        "contribution",
                        "technical_driver",
                    ]
                ]
                .sort_values(
                    "contribution",
                    ascending=False,
                )
            )

            driver_summary = (
                driver_summary.rename(
                    columns={
                        "driver":
                        "Driver",
                        "level":
                        "Level",
                        "contribution":
                        "Average Score",
                        "technical_driver":
                        "Technical Field",
                    }
                )
            )

            st.dataframe(
                driver_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Average Score":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    )
                },
            )

        else:
            st.caption(
                "No driver data is available "
                "for the selected risk."
            )

        st.divider()

        # =====================================================
        # Top events driving signal
        # =====================================================

        st.markdown(
            "### Top Events Driving Signal"
        )

        if risk_events.empty:
            st.caption(
                "No events were found for this "
                "risk category using the selected filters."
            )

        else:
            if (
                "Published_utc"
                in risk_events.columns
            ):
                risk_events[
                    "Published_utc"
                ] = pd.to_datetime(
                    risk_events[
                        "Published_utc"
                    ],
                    errors="coerce",
                )

            else:
                risk_events[
                    "Published_utc"
                ] = pd.NaT

            risk_events[
                "Event_Label_Group"
            ] = risk_events.apply(
                lambda row: clean_event_label(
                    row.get(
                        "Event_Label"
                    ),
                    row.get(
                        "Title"
                    ),
                ),
                axis=1,
            )

            grouped_events = (
                risk_events
                .groupby(
                    "Event_Label_Group",
                    dropna=False,
                )
                .agg(
                    Raw_Event_Label=(
                        "Event_Label",
                        "first",
                    ),
                    Event_Severity=(
                        "Event_Severity",
                        "max",
                    ),
                    Article_Count=(
                        "Title",
                        "nunique",
                    ),
                    Latest_Date=(
                        "Published_utc",
                        "max",
                    ),
                    Sample_Title=(
                        "Title",
                        "first",
                    ),
                    Sample_Content=(
                        "Content",
                        "first",
                    ),
                    Sample_Link=(
                        "Link",
                        "first",
                    ),
                )
                .reset_index()
                .sort_values(
                    "Event_Severity",
                    ascending=False,
                )
            )

            for _, event in (
                grouped_events.iterrows()
            ):
                event_label = event.get(
                    "Event_Label_Group"
                )

                sev = (
                    float(
                        event[
                            "Event_Severity"
                        ]
                    )
                    if pd.notna(
                        event[
                            "Event_Severity"
                        ]
                    )
                    else 0
                )

                sev_tag = (
                    severity_bucket(
                        sev
                    )
                )

                with st.expander(
                    (
                        f"{event_label} — "
                        f"Severity {sev:.2f} "
                        f"({sev_tag})"
                    ),
                    expanded=False,
                ):
                    article_count = int(
                        event[
                            "Article_Count"
                        ]
                    )

                    st.caption(
                        (
                            f"{article_count} "
                            "related article(s)"
                        )
                    )

                    sample_title = (
                        event.get(
                            "Sample_Title"
                        )
                    )

                    if (
                        pd.notna(
                            sample_title
                        )
                        and str(
                            sample_title
                        ).strip()
                    ):
                        st.markdown(
                            (
                                "**Representative headline:** "
                                f"{sample_title}"
                            )
                        )

                    sample_content = (
                        event.get(
                            "Sample_Content"
                        )
                    )

                    if (
                        pd.notna(
                            sample_content
                        )
                        and str(
                            sample_content
                        ).strip()
                    ):
                        content_text = str(
                            sample_content
                        )

                        preview = (
                            content_text[:700]
                        )

                        st.write(
                            preview
                            + (
                                "..."
                                if len(
                                    content_text
                                ) > 700
                                else ""
                            )
                        )

                    sample_link = str(
                        event.get(
                            "Sample_Link",
                            "",
                        )
                    ).strip()

                    if sample_link.startswith(
                        "http"
                    ):
                        st.markdown(
                            (
                                "[Read representative "
                                f"article]({sample_link})"
                            )
                        )

                    else:
                        st.caption(
                            "No representative "
                            "article link is available."
                        )

                    raw_event_label = (
                        event.get(
                            "Raw_Event_Label"
                        )
                    )

                    missing_raw_label = (
                        pd.isna(
                            raw_event_label
                        )
                        or str(
                            raw_event_label
                        )
                        .strip()
                        .lower()
                        in [
                            "",
                            "nan",
                            "none",
                        ]
                    )

                    if missing_raw_label:
                        related = (
                            risk_events[
                                risk_events[
                                    "Title"
                                ]
                                == sample_title
                            ]
                            .copy()
                        )

                    else:
                        related = (
                            risk_events[
                                risk_events[
                                    "Event_Label"
                                ]
                                == raw_event_label
                            ]
                            .copy()
                        )

                    st.markdown(
                        "**Related articles**"
                    )

                    related_dedup_columns = [
                        column
                        for column in [
                            "Title",
                            "Link",
                        ]
                        if column
                        in related.columns
                    ]

                    if related_dedup_columns:
                        related = (
                            related
                            .drop_duplicates(
                                subset=(
                                    related_dedup_columns
                                )
                            )
                        )

                    for _, article_row in (
                        related
                        .head(
                            max_items
                        )
                        .iterrows()
                    ):
                        title = (
                            article_row.get(
                                "Title"
                            )
                        )

                        if (
                            pd.isna(title)
                            or not str(
                                title
                            ).strip()
                        ):
                            continue

                        published = (
                            article_row.get(
                                "Published_utc"
                            )
                        )

                        link = str(
                            article_row.get(
                                "Link",
                                "",
                            )
                        ).strip()

                        source = (
                            article_row.get(
                                "Source",
                                article_row.get(
                                    "source",
                                    article_row.get(
                                        "canonical_source",
                                        (
                                            "Unknown "
                                            "source"
                                        ),
                                    ),
                                ),
                            )
                        )

                        with st.container(
                            border=True
                        ):
                            if link.startswith(
                                "http"
                            ):
                                st.markdown(
                                    f"**[{title}]({link})**"
                                )

                            else:
                                st.markdown(
                                    f"**{title}**"
                                )

                            meta = []

                            if (
                                pd.notna(source)
                                and str(
                                    source
                                ).strip()
                            ):
                                meta.append(
                                    str(source)
                                )

                            if pd.notna(
                                published
                            ):
                                published_value = (
                                    pd.to_datetime(
                                        published,
                                        errors="coerce",
                                    )
                                )

                                if pd.notna(
                                    published_value
                                ):
                                    meta.append(
                                        (
                                            "Published: "
                                            f"{published_value:%B %d, %Y}"
                                        )
                                    )

                            if meta:
                                st.caption(
                                    " | ".join(
                                        meta
                                    )
                                )

                            content = (
                                article_row.get(
                                    "Content"
                                )
                            )

                            if (
                                pd.notna(
                                    content
                                )
                                and str(
                                    content
                                ).strip()
                            ):
                                content_text = (
                                    str(content)
                                )

                                content_preview = (
                                    content_text[
                                        :450
                                    ]
                                )

                                st.write(
                                    content_preview
                                    + (
                                        "..."
                                        if len(
                                            content_text
                                        ) > 450
                                        else ""
                                    )
                                )

                            if not link.startswith(
                                "http"
                            ):
                                st.caption(
                                    "Original article "
                                    "link unavailable."
                                )

        st.divider()

        # =====================================================
        # Recent headlines
        # =====================================================

        st.markdown(
            "### Recent Headlines"
        )

        headline_df = (
            risk_events.copy()
        )

        date_col = (
            "Published_utc"
            if (
                "Published_utc"
                in headline_df.columns
            )
            else "Window"
        )

        if date_col in headline_df.columns:
            headline_df[
                date_col
            ] = pd.to_datetime(
                headline_df[
                    date_col
                ],
                errors="coerce",
            )

        headline_dedup_columns = [
            column
            for column in [
                "Title",
                "Link",
            ]
            if column
            in headline_df.columns
        ]

        recent_headlines = (
            headline_df
            .dropna(
                subset=["Title"]
            )
            .sort_values(
                date_col,
                ascending=False,
            )
        )

        if headline_dedup_columns:
            recent_headlines = (
                recent_headlines
                .drop_duplicates(
                    subset=(
                        headline_dedup_columns
                    )
                )
            )

        recent_headlines = (
            recent_headlines
            .head(
                max_items
            )
        )

        if recent_headlines.empty:
            st.caption(
                "No recent headlines are "
                "available for the selected risk."
            )

        else:
            for _, headline in (
                recent_headlines.iterrows()
            ):
                headline_title = str(
                    headline.get(
                        "Title",
                        "Untitled headline",
                    )
                )

                with st.expander(
                    headline_title[:180],
                    expanded=False,
                ):
                    if pd.notna(
                        headline.get(
                            date_col
                        )
                    ):
                        headline_date = (
                            pd.to_datetime(
                                headline.get(
                                    date_col
                                ),
                                errors="coerce",
                            )
                        )

                        if pd.notna(
                            headline_date
                        ):
                            st.caption(
                                (
                                    "Published: "
                                    f"{headline_date:%B %d, %Y}"
                                )
                            )

                    source = headline.get(
                        "Source",
                        headline.get(
                            "source",
                            headline.get(
                                "canonical_source",
                                None,
                            ),
                        ),
                    )

                    if (
                        pd.notna(source)
                        and str(
                            source
                        ).strip()
                    ):
                        st.caption(
                            f"Source: {source}"
                        )

                    content = (
                        headline.get(
                            "Content"
                        )
                    )

                    if (
                        pd.notna(content)
                        and str(
                            content
                        ).strip()
                    ):
                        content_text = str(
                            content
                        )

                        st.write(
                            content_text[:700]
                            + (
                                "..."
                                if len(
                                    content_text
                                ) > 700
                                else ""
                            )
                        )

                    link = str(
                        headline.get(
                            "Link",
                            "",
                        )
                    ).strip()

                    if link.startswith(
                        "http"
                    ):
                        st.markdown(
                            (
                                "[Read original "
                                f"article]({link})"
                            )
                        )

                    else:
                        st.caption(
                            "Original article "
                            "link unavailable."
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
    # Manual review overrides are durable state. GCS is the source of truth;
    # GitHub remains an audit mirror only.
    ARTICLE_CHANGES_BLOB = (
        "latest/manual_overrides/BERTopic_changes.csv"
    )
    STORY_CHANGES_BLOB = (
        "latest/manual_overrides/Story_changes.csv"
    )
    change_log_path = Path(
        "Model_training/BERTopic_changes.csv"
    )
    story_change_log = Path(
        "Model_training/Story_changes.csv"
    )

    ARTICLE_OVERRIDE_MAP = {
        "Predicted_Risks_Upd": "Predicted_Risks_new",
        "Recency_Upd": "Recency",
        "Acceleration_value_Upd": "Acceleration_value",
        "Source_Accuracy_Upd": "Source_Accuracy",
        "Impact_Score_Upd": "Impact_Score",
        "Location_Upd": "Location",
        "Industry_Risk_Upd": "Industry_Risk",
        "Frequency_Score_Upd": "Frequency_Score",
    }

    STORY_OVERRIDE_MAP = {
        "Predicted_Risks_Upd": "risk_label",
        "Recency_Upd": "avg_recency",
        "Acceleration_value_Upd": "avg_acceleration",
        "Source_Accuracy_Upd": "avg_source_accuracy",
        "Impact_Score_Upd": "avg_impact_score",
        "Location_Upd": "avg_location",
        "Industry_Risk_Upd": "avg_industry_risk",
        "Frequency_Score_Upd": "avg_frequency",
    }

    OVERRIDE_META_COLUMNS = [
        "Reviewed",
        "Reviewed_at",
        "Changed_at",
        "Change reason",
    ]

    def save_override_log(
        data: pd.DataFrame,
        local_path: Path,
        blob_path: str,
    ) -> None:
        local_path = Path(local_path)
        local_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = local_path.with_suffix(
            local_path.suffix + ".tmp"
        )

        data.to_csv(
            temp_path,
            index=False,
        )

        client = get_gcs_client()
        bucket = client.bucket(
            "tulane-risk-data"
        )
        blob = bucket.blob(
            blob_path
        )
        blob.upload_from_filename(
            str(temp_path),
            content_type="text/csv",
        )

        os.replace(
            temp_path,
            local_path,
        )

    def load_override_log(
        blob_path: str,
        local_path: Path,
    ) -> pd.DataFrame:
        local_path = Path(local_path)
        local_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if blob_exists(
            blob_path,
            bucket_name="tulane-risk-data",
        ):
            download_blob(
                blob_path,
                str(local_path),
            )

            if (
                local_path.exists()
                and local_path.stat().st_size > 0
            ):
                return pd.read_csv(
                    local_path,
                    low_memory=False,
                )

        # One-time migration path: if GCS has not been seeded yet,
        # use the repository copy and immediately persist it.
        if (
            local_path.exists()
            and local_path.stat().st_size > 0
        ):
            migrated = pd.read_csv(
                local_path,
                low_memory=False,
            )

            save_override_log(
                migrated,
                local_path,
                blob_path,
            )

            return migrated

        return pd.DataFrame()

    def append_override_row(
        row: dict,
        blob_path: str,
        local_path: Path,
    ) -> pd.DataFrame:
        """Append safely without silently overwriting a concurrent edit."""
        from google.api_core.exceptions import PreconditionFailed

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        client = get_gcs_client()
        bucket = client.bucket("tulane-risk-data")

        for attempt in range(5):
            blob = bucket.blob(blob_path)

            if blob.exists(client=client):
                blob.reload(client=client)
                generation = int(blob.generation)
                payload = blob.download_as_bytes(
                    if_generation_match=generation
                )

                if payload:
                    try:
                        current = pd.read_csv(
                            io.BytesIO(payload),
                            low_memory=False,
                        )
                    except pd.errors.EmptyDataError:
                        current = pd.DataFrame()
                else:
                    current = pd.DataFrame()
            else:
                generation = 0

                # First deployment: seed from the existing repository copy.
                if local_path.exists() and local_path.stat().st_size > 0:
                    current = pd.read_csv(local_path, low_memory=False)
                else:
                    current = pd.DataFrame()

            updated = pd.concat(
                [current, pd.DataFrame([row])],
                ignore_index=True,
                sort=False,
            )

            csv_text = updated.to_csv(index=False)

            try:
                blob.upload_from_string(
                    csv_text,
                    content_type="text/csv",
                    if_generation_match=generation,
                )
            except PreconditionFailed:
                # Another administrator saved between our read and write.
                # Reload that newer object and retry the append.
                time.sleep(0.25 * (attempt + 1))
                continue

            temp_path = local_path.with_suffix(local_path.suffix + ".tmp")
            temp_path.write_text(csv_text, encoding="utf-8")
            os.replace(temp_path, local_path)
            return updated

        raise RuntimeError(
            "The override log changed repeatedly while saving. "
            "Please submit the edit again."
        )

    def clean_single_risk(value):
        if isinstance(value, list):
            return (
                str(value[0]).strip()
                if value
                else "No Risk"
            )

        if pd.isna(value):
            return "No Risk"

        text = str(value).strip()

        if (
            text.startswith("[")
            and text.endswith("]")
        ):
            for parser_func in (
                json.loads,
                ast.literal_eval,
            ):
                try:
                    parsed = parser_func(text)

                    if isinstance(parsed, list):
                        return (
                            str(parsed[0]).strip()
                            if parsed
                            else "No Risk"
                        )

                except Exception:
                    continue

        return text or "No Risk"

    def normalize_story_id(value):
        if pd.isna(value):
            return ""

        text = str(value).strip()

        if re.fullmatch(
            r"-?\d+\.0",
            text,
        ):
            text = text[:-2]

        return text

    def _normalize_override_url(value):
        if pd.isna(value):
            return ""

        text = str(value).strip()

        if not text:
            return ""

        if not text.casefold().startswith(
            ("http://", "https://")
        ):
            return text.casefold().rstrip("/")

        try:
            parsed = urlsplit(text)
            query = []

            for key, query_value in parse_qsl(
                parsed.query,
                keep_blank_values=True,
            ):
                key_lower = key.casefold()

                if (
                    key_lower.startswith("utm_")
                    or key_lower
                    in {
                        "fbclid",
                        "gclid",
                        "mc_cid",
                        "mc_eid",
                    }
                ):
                    continue

                query.append((key, query_value))

            normalized_path = re.sub(
                r"/+$",
                "",
                parsed.path or "",
            )

            return urlunsplit(
                (
                    parsed.scheme.casefold(),
                    parsed.netloc.casefold(),
                    normalized_path,
                    urlencode(sorted(query)),
                    "",
                )
            )

        except ValueError:
            return text.casefold().rstrip("/")

    TRACKING_QUERY_KEYS = {
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
    }

    def normalize_override_url(value):
        if value is None or pd.isna(value):
            return ""

        text = str(value).strip()
        if not text:
            return ""

        if not text.casefold().startswith(("http://", "https://")):
            return text.casefold().rstrip("/")

        try:
            parsed = urlsplit(text)
            cleaned_query = []

            for key, query_value in parse_qsl(
                parsed.query,
                keep_blank_values=True,
            ):
                key_lower = key.casefold()
                if (
                    key_lower.startswith("utm_")
                    or key_lower in TRACKING_QUERY_KEYS
                ):
                    continue
                cleaned_query.append((key, query_value))

            normalized_path = re.sub(
                r"/+$",
                "",
                parsed.path or "",
            )

            return urlunsplit(
                (
                    parsed.scheme.casefold(),
                    parsed.netloc.casefold(),
                    normalized_path,
                    urlencode(sorted(cleaned_query)),
                    "",
                )
            )

        except ValueError:
            return text.casefold().rstrip("/")

    def article_override_key(data):
        links = data.get(
            "Link",
            pd.Series("", index=data.index),
        ).apply(normalize_override_url)

        titles = (
            data.get(
                "Title",
                pd.Series("", index=data.index),
            )
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.casefold()
        )

        if "Published_utc" in data.columns:
            published_source = data["Published_utc"]
        else:
            published_source = data.get(
                "Published",
                pd.Series("", index=data.index),
            )

        published = pd.to_datetime(
            published_source,
            errors="coerce",
            utc=True,
        ).astype(str)
        published = published.where(
            ~published.eq("NaT"),
            "",
        )

        fallback = (
            "title::"
            + titles
            + "|published::"
            + published
        )

        return links.where(
            links.ne(""),
            fallback,
        )

    def _blank_override_value(value):
        if value is None:
            return True

        try:
            if pd.isna(value):
                return True
        except (TypeError, ValueError):
            pass

        return str(value).strip().casefold() in {
            "",
            "nan",
            "none",
            "<na>",
        }

    def _collapse_override_history(
        changes: pd.DataFrame,
        key_column: str,
        value_columns: list[str],
    ) -> pd.DataFrame:
        """Keep the newest nonblank value for each editable field."""
        if (
            changes is None
            or changes.empty
            or key_column not in changes.columns
        ):
            return pd.DataFrame(
                columns=[key_column, *value_columns]
            )

        work = changes.copy()
        work["_override_row_order"] = range(len(work))
        work["_override_changed_at"] = pd.to_datetime(
            work.get("Changed_at"),
            errors="coerce",
            utc=True,
        )
        work = work[
            work[key_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        ].copy()

        if work.empty:
            return pd.DataFrame(
                columns=[key_column, *value_columns]
            )

        work = work.sort_values(
            ["_override_changed_at", "_override_row_order"],
            ascending=[True, True],
            na_position="first",
        )

        records = []
        available = [
            column
            for column in value_columns
            if column in work.columns
        ]

        for key, group in work.groupby(
            key_column,
            sort=False,
            dropna=False,
        ):
            record = {key_column: key}

            for column in available:
                valid = ~group[column].apply(
                    _blank_override_value
                )

                if valid.any():
                    record[column] = (
                        group.loc[valid, column].iloc[-1]
                    )

            records.append(record)

        return pd.DataFrame(records)

    def apply_latest_overrides(
        base_data: pd.DataFrame,
        changes: pd.DataFrame,
        item_type: str,
    ) -> pd.DataFrame:
        result = base_data.copy()

        if item_type == "article":
            result["_override_key"] = (
                article_override_key(result)
            )
            field_map = ARTICLE_OVERRIDE_MAP

        elif item_type == "story":
            story_ids = (
                result.get(
                    "story_id",
                    pd.Series(
                        "",
                        index=result.index,
                    ),
                )
                .apply(normalize_story_id)
            )

            result["_override_key"] = (
                story_ids.apply(
                    lambda value: (
                        f"story::{value}"
                        if value
                        else ""
                    )
                )
            )
            field_map = STORY_OVERRIDE_MAP

        else:
            raise ValueError(
                f"Unsupported override item type: {item_type}"
            )

        for update_column in field_map:
            if update_column not in result.columns:
                result[update_column] = pd.NA

        for column in OVERRIDE_META_COLUMNS:
            if column not in result.columns:
                if column == "Reviewed":
                    result[column] = 0
                elif column in {
                    "Reviewed_at",
                    "Changed_at",
                }:
                    result[column] = pd.NaT
                else:
                    result[column] = ""

        if changes is None or changes.empty:
            return result.drop(
                columns=["_override_key"],
                errors="ignore",
            )

        history = changes.copy()

        if item_type == "article":
            history["_override_key"] = (
                article_override_key(history)
            )
        else:
            story_ids = (
                history.get(
                    "story_id",
                    pd.Series(
                        "",
                        index=history.index,
                    ),
                )
                .apply(normalize_story_id)
            )
            history["_override_key"] = (
                story_ids.apply(
                    lambda value: (
                        f"story::{value}"
                        if value
                        else ""
                    )
                )
            )

        latest = _collapse_override_history(
            history,
            "_override_key",
            list(field_map.keys())
            + OVERRIDE_META_COLUMNS,
        )

        if latest.empty:
            return result.drop(
                columns=["_override_key"],
                errors="ignore",
            )

        result = result.merge(
            latest,
            on="_override_key",
            how="left",
            suffixes=("", "__manual"),
        )

        manual_factor_mask = pd.Series(
            False,
            index=result.index,
        )

        for update_column, base_column in (
            field_map.items()
        ):
            manual_column = (
                f"{update_column}__manual"
            )

            if manual_column not in result.columns:
                continue

            manual_values = result[manual_column]

            if update_column == "Predicted_Risks_Upd":
                manual_values = manual_values.apply(
                    lambda value: (
                        clean_single_risk(value)
                        if not _blank_override_value(value)
                        else pd.NA
                    )
                )
            else:
                manual_values = pd.to_numeric(
                    manual_values,
                    errors="coerce",
                )
                manual_factor_mask = (
                    manual_factor_mask
                    | manual_values.notna()
                )

            result[update_column] = (
                manual_values.combine_first(
                    result[update_column]
                )
            )

            if base_column not in result.columns:
                result[base_column] = pd.NA

            result[base_column] = (
                manual_values.combine_first(
                    result[base_column]
                )
            )

        for column in OVERRIDE_META_COLUMNS:
            manual_column = f"{column}__manual"

            if manual_column in result.columns:
                result[column] = (
                    result[manual_column]
                    .combine_first(result[column])
                )

        result["Reviewed"] = (
            pd.to_numeric(
                result["Reviewed"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )
        result.loc[
            result["Reviewed"].eq(0),
            "Reviewed_at",
        ] = pd.NaT

        if item_type == "article":
            weights = {
                "Recency": 0.15,
                "Source_Accuracy": 0.10,
                "Impact_Score": 0.35,
                "Acceleration_value": 0.25,
                "Location": 0.05,
                "Industry_Risk": 0.05,
                "Frequency_Score": 0.05,
            }

            # Leave untouched pipeline scores exactly as they were.
            # Recalculate only records with a saved manual factor.
            if manual_factor_mask.any():
                score = pd.Series(
                    0.0,
                    index=result.index[manual_factor_mask],
                )

                for column, weight in weights.items():
                    values = pd.to_numeric(
                        result.loc[manual_factor_mask, column],
                        errors="coerce",
                    ).fillna(0.0)
                    score = score + (values * weight)

                result.loc[
                    manual_factor_mask,
                    "Risk_Score",
                ] = score / sum(weights.values())

        else:
            if "risk_label" in result.columns:
                result["Predicted_Risks_new"] = (
                    result["risk_label"]
                )

            story_weights = {
                "avg_recency": 0.15,
                "avg_source_accuracy": 0.10,
                "avg_impact_score": 0.35,
                "avg_acceleration": 0.25,
                "avg_location": 0.05,
                "avg_industry_risk": 0.05,
                "avg_frequency": 0.05,
            }

            if manual_factor_mask.any():
                score = pd.Series(
                    0.0,
                    index=result.index[manual_factor_mask],
                )

                for column, weight in story_weights.items():
                    values = pd.to_numeric(
                        result.loc[manual_factor_mask, column],
                        errors="coerce",
                    ).fillna(0.0)
                    score = score + (values * weight)

                result.loc[
                    manual_factor_mask,
                    "avg_risk_score",
                ] = score / sum(story_weights.values())

        columns_to_drop = [
            column
            for column in result.columns
            if column.endswith("__manual")
        ] + ["_override_key"]

        return result.drop(
            columns=columns_to_drop,
            errors="ignore",
        )

    def apply_story_overrides_to_article_rows(
        base_data: pd.DataFrame,
        changes: pd.DataFrame,
    ) -> pd.DataFrame:
        """Show a saved story edit on every constituent article immediately."""
        result = base_data.copy()

        if (
            changes is None
            or changes.empty
            or "story_id" not in result.columns
            or "story_id" not in changes.columns
        ):
            return result

        result["_override_key"] = (
            result["story_id"]
            .apply(normalize_story_id)
            .apply(lambda value: f"story::{value}" if value else "")
        )

        history = changes.copy()
        history["_override_key"] = (
            history["story_id"]
            .apply(normalize_story_id)
            .apply(lambda value: f"story::{value}" if value else "")
        )

        latest = _collapse_override_history(
            history,
            "_override_key",
            list(ARTICLE_OVERRIDE_MAP.keys()),
        )

        if latest.empty:
            return result.drop(columns=["_override_key"], errors="ignore")

        result = result.merge(
            latest,
            on="_override_key",
            how="left",
            suffixes=("", "__story_manual"),
        )

        factor_mask = pd.Series(False, index=result.index)

        for update_column, base_column in ARTICLE_OVERRIDE_MAP.items():
            manual_column = f"{update_column}__story_manual"
            if manual_column not in result.columns:
                continue

            if update_column == "Predicted_Risks_Upd":
                values = result[manual_column].apply(
                    lambda value: (
                        clean_single_risk(value)
                        if not _blank_override_value(value)
                        else pd.NA
                    )
                )
                valid = values.notna()
            else:
                values = pd.to_numeric(
                    result[manual_column],
                    errors="coerce",
                )
                valid = values.notna()
                factor_mask = factor_mask | valid

            if update_column not in result.columns:
                result[update_column] = pd.NA
            if base_column not in result.columns:
                result[base_column] = pd.NA

            result.loc[valid, update_column] = values.loc[valid]
            result.loc[valid, base_column] = values.loc[valid]

        if factor_mask.any():
            weights = {
                "Recency": 0.15,
                "Source_Accuracy": 0.10,
                "Impact_Score": 0.35,
                "Acceleration_value": 0.25,
                "Location": 0.05,
                "Industry_Risk": 0.05,
                "Frequency_Score": 0.05,
            }
            score = pd.Series(0.0, index=result.index[factor_mask])
            for column, weight in weights.items():
                score = score + (
                    pd.to_numeric(
                        result.loc[factor_mask, column],
                        errors="coerce",
                    ).fillna(0.0)
                    * weight
                )
            result.loc[factor_mask, "Risk_Score"] = (
                score / sum(weights.values())
            )

        return result.drop(
            columns=[
                column
                for column in result.columns
                if column.endswith("__story_manual")
            ]
            + ["_override_key"],
            errors="ignore",
        )

    def clear_review_session_state():
        for key in (
            "articles",
            "change_log",
            "story_log",
        ):
            st.session_state.pop(
                key,
                None,
            )
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

    if "articles" not in st.session_state:
        try:
            results_df = load_csv_gz_from_gcs(
                "latest/topics/BERTopic_Streamlit.csv.gz",
                "pipeline/resources/BERTopic_Streamlit.csv.gz",
            )

            article_changes = load_override_log(
                ARTICLE_CHANGES_BLOB,
                change_log_path,
            )

            st.session_state.change_log = (
                article_changes
            )

            st.session_state.articles = (
                apply_latest_overrides(
                    base_data=results_df,
                    changes=article_changes,
                    item_type="article",
                )
            )

        except Exception as error:
            st.error(
                "Failed to load BERTopic results "
                f"and manual overrides: {error}"
            )
            st.stop()

    stories = load_csv_gz_from_gcs(
        "latest/dashboard/dashboard_stories.csv.gz",
        "pipeline/resources/dashboard_stories.csv.gz",
    )

    stories = stories[
        ~stories["canonical_title"]
        .str.match(
            r"^Story \d+",
            na=False,
        )
    ].copy()

    story_changes = load_override_log(
        STORY_CHANGES_BLOB,
        story_change_log,
    )

    st.session_state.story_log = (
        story_changes
    )

    stories = apply_latest_overrides(
        base_data=stories,
        changes=story_changes,
        item_type="story",
    )

    dropdown = load_csv_gz_from_gcs(
        "latest/dashboard/dashboard_dropdown.csv.gz",
        "pipeline/resources/dashboard_dropdown.csv.gz",
    )

    # Apply article edits first and story edits second so the story view
    # immediately shows the same approved values as the story card.
    dropdown = apply_latest_overrides(
        base_data=dropdown,
        changes=st.session_state.change_log,
        item_type="article",
    )
    dropdown = apply_story_overrides_to_article_rows(
        dropdown,
        story_changes,
    )

    for column in numeric_cols:
        if column in st.session_state.articles.columns:
            st.session_state.articles[
                column
            ] = pd.to_numeric(
                st.session_state.articles[
                    column
                ],
                errors="coerce",
            )

    
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
    if "Reviewed" not in stories_timeline.columns:
        stories_timeline["Reviewed"] = 0
    else:
        stories_timeline["Reviewed"] = (
            pd.to_numeric(
                stories_timeline["Reviewed"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

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

    saved_notice = st.session_state.pop(
        "review_save_notice",
        None,
    )

    if saved_notice:
        st.success(saved_notice)

    #give me a filter to filter articles by date range
    st.sidebar.header("Filter Articles")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=30))
    end_date = st.sidebar.date_input("End Date", datetime.now())


    if start_date > end_date:
        st.sidebar.error("Start date must be before end date.")
    # Load articles and risks


    update_cols = ['Predicted_Risks_Upd', 'Recency_Upd', 'Acceleration_value_Upd', 'Source_Accuracy_Upd',
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
                            base_row = story.to_dict()
                            base_row[
                                "Predicted_Risks_Upd"
                            ] = (
                                selected_risks[0]
                                if selected_risks
                                else "No Risk"
                            )
                            base_row[
                                "Recency_Upd"
                            ] = upd_recency_value
                            base_row[
                                "Acceleration_value_Upd"
                            ] = upd_acceleration_value
                            base_row[
                                "Source_Accuracy_Upd"
                            ] = upd_source_accuracy
                            base_row[
                                "Impact_Score_Upd"
                            ] = upd_impact_score
                            base_row[
                                "Location_Upd"
                            ] = upd_location
                            base_row[
                                "Industry_Risk_Upd"
                            ] = upd_industry_risk
                            base_row[
                                "Frequency_Score_Upd"
                            ] = upd_frequency_score
                            base_row[
                                "Change reason"
                            ] = reason
                            base_row[
                                "Changed_at"
                            ] = pd.Timestamp.now(
                                tz="UTC"
                            )
                            base_row["Reviewed"] = 1
                            base_row[
                                "Reviewed_at"
                            ] = pd.Timestamp.now(
                                tz="UTC"
                            )

                            try:
                                st.session_state.story_log = (
                                    append_override_row(
                                        row=base_row,
                                        blob_path=(
                                            STORY_CHANGES_BLOB
                                        ),
                                        local_path=(
                                            story_change_log
                                        ),
                                    )
                                )

                                try:
                                    push_file_to_github(
                                        story_change_log,
                                        repo=(
                                            "ERSRisk/"
                                            "tulane-sentiment-app-clean"
                                        ),
                                        dest_path=(
                                            "Model_training/"
                                            "Story_changes.csv"
                                        ),
                                        branch="main",
                                    )

                                except Exception as github_error:
                                    st.warning(
                                        "The change is saved "
                                        "permanently in GCS, but "
                                        "the GitHub audit mirror "
                                        "could not be updated: "
                                        f"{github_error}"
                                    )

                                clear_review_session_state()

                                st.session_state[
                                    "review_save_notice"
                                ] = (
                                    "Story changes saved "
                                    "permanently."
                                )

                                st.rerun()

                            except Exception as error:
                                st.error(
                                    "The story change could "
                                    "not be saved to GCS: "
                                    f"{error}"
                                )
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
                                for override_field in ARTICLE_OVERRIDE_MAP:
                                    new_row[override_field] = pd.NA
                                new_row["Change reason"] = pd.NA
                                new_row["Reviewed"] = 1
                                new_row[
                                    "Reviewed_at"
                                ] = pd.Timestamp.now(
                                    tz="UTC"
                                )
                                new_row[
                                    "Changed_at"
                                ] = pd.Timestamp.now(
                                    tz="UTC"
                                )

                                st.session_state.change_log = (
                                    append_override_row(
                                        row=new_row,
                                        blob_path=(
                                            ARTICLE_CHANGES_BLOB
                                        ),
                                        local_path=(
                                            change_log_path
                                        ),
                                    )
                                )

                                clear_review_session_state()

                                st.session_state[
                                    "review_save_notice"
                                ] = (
                                    "Article marked as "
                                    "reviewed permanently."
                                )

                                st.rerun()
                        else:
                            if st.button("Unmark reviewed", key=f"unmark_{row_id}"):
                                new_row = article.to_dict()
                                for override_field in ARTICLE_OVERRIDE_MAP:
                                    new_row[override_field] = pd.NA
                                new_row["Change reason"] = pd.NA
                                new_row["Reviewed"] = 0
                                new_row["Reviewed_at"] = pd.NaT
                                new_row[
                                    "Changed_at"
                                ] = pd.Timestamp.now(
                                    tz="UTC"
                                )

                                st.session_state.change_log = (
                                    append_override_row(
                                        row=new_row,
                                        blob_path=(
                                            ARTICLE_CHANGES_BLOB
                                        ),
                                        local_path=(
                                            change_log_path
                                        ),
                                    )
                                )

                                clear_review_session_state()

                                st.session_state[
                                    "review_save_notice"
                                ] = (
                                    "Article review mark "
                                    "removed permanently."
                                )

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
                                    new_row = article.to_dict()

                                    new_row[
                                        "Predicted_Risks_Upd"
                                    ] = (
                                        selected_risks[0]
                                        if selected_risks
                                        else "No Risk"
                                    )
                                    new_row[
                                        "Recency_Upd"
                                    ] = upd_recency_value
                                    new_row[
                                        "Acceleration_value_Upd"
                                    ] = upd_acceleration_value
                                    new_row[
                                        "Source_Accuracy_Upd"
                                    ] = upd_source_accuracy
                                    new_row[
                                        "Impact_Score_Upd"
                                    ] = upd_impact_score
                                    new_row[
                                        "Location_Upd"
                                    ] = upd_location
                                    new_row[
                                        "Industry_Risk_Upd"
                                    ] = upd_industry_risk
                                    new_row[
                                        "Frequency_Score_Upd"
                                    ] = upd_frequency_score
                                    new_row[
                                        "Change reason"
                                    ] = reason
                                    new_row[
                                        "Changed_at"
                                    ] = pd.Timestamp.now(
                                        tz="UTC"
                                    )
                                    new_row["Reviewed"] = 1
                                    new_row[
                                        "Reviewed_at"
                                    ] = pd.Timestamp.now(
                                        tz="UTC"
                                    )

                                    try:
                                        st.session_state.change_log = (
                                            append_override_row(
                                                row=new_row,
                                                blob_path=(
                                                    ARTICLE_CHANGES_BLOB
                                                ),
                                                local_path=(
                                                    change_log_path
                                                ),
                                            )
                                        )

                                        try:
                                            push_file_to_github(
                                                change_log_path,
                                                repo=(
                                                    "ERSRisk/"
                                                    "tulane-sentiment-app-clean"
                                                ),
                                                dest_path=(
                                                    "Model_training/"
                                                    "BERTopic_changes.csv"
                                                ),
                                                branch="main",
                                            )

                                        except Exception as github_error:
                                            st.warning(
                                                "The change is saved "
                                                "permanently in GCS, "
                                                "but the GitHub audit "
                                                "mirror could not be "
                                                "updated: "
                                                f"{github_error}"
                                            )

                                        clear_review_session_state()

                                        st.session_state[
                                            "review_save_notice"
                                        ] = (
                                            "Article risk and "
                                            "factor changes saved "
                                            "permanently."
                                        )

                                        st.rerun()

                                    except Exception as error:
                                        st.error(
                                            "The article change "
                                            "could not be saved "
                                            "to GCS: "
                                            f"{error}"
                                        )
   
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
