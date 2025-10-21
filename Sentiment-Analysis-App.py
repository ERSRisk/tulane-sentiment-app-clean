import streamlit as st
import requests
from google import genai
import pandas as pd
import re
from datetime import timedelta, datetime
from streamlit_tags import st_tags
import plotly.express as px
from newspaper import Article
from google.genai.errors import ClientError
import random
import time
import asyncio
from lxml.html.clean import Cleaner
from dateutil import parser
from textblob import TextBlob
import tweepy
import nest_asyncio
import json
import re
import altair as alt
import matplotlib.pyplot as plt
import base64
import io
import os
import tempfile
from typing import Iterable, Any
from zoneinfo import ZoneInfo
import gzip



st.set_page_config(page_title="Tulane Risk Dashboard")
st.sidebar.title("Navigation")
st.sidebar.markdown("Select a tool:")
selection = st.sidebar.selectbox("Choose a tool:", ["Article Risk Review", "Unmatched Topic Analysis", "Risk/Event Detector"])

if "current_tab" not in st.session_state:
    st.session_state.current_tab = selection

# If switching tabs, clear session except the current tab
if st.session_state.current_tab != selection:
    keys_to_keep = {"current_tab"}
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    st.session_state.current_tab = selection

if selection == "Unmatched Topic Analysis":
    def push_file_to_github(local_path:str, repo:str, dest_path:str, branch:str = "main", token:str|None = None):
        token = st.secrets['all_my_api_keys']['GITHUB_TOKEN']
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
            st.secrets['all_my_api_keys']['GITHUB_TOKEN']
            ) or []

    if 'topicsbert' not in st.session_state:
        if os.path.exists('Model_training/topics_BERT.json'):
            with open('Model_training/topics_BERT.json', 'r') as f:
                st.session_state.topicsbert = json.load(f)
        else:
            st.session_state.topicsbert = []
    
    if 'discarded' not in st.session_state:
        st.session_state.discarded = fetch_release(
            "ERSRisk", "tulane-sentiment-app-clean",
            "discarded-topics", "discarded_topics.json",
            st.secrets['all_my_api_keys']['GITHUB_TOKEN']
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
                        next_id = next_topic_id(saved_ids, start = 0)
                        new_topic = {
                            'topic': next_id,
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
                        
                        # Update the single canonical unmatched file (no Contents API!)
                        resp6 = upsert_single_big_json(
                            owner="ERSRisk",
                            repo="tulane-sentiment-app-clean",
                            tag="unmatched-topics",
                            asset_name="unmatched_topics.json",
                            new_items=st.session_state.unmatched,
                            dedupe_key="topic",
                            token=st.secrets['all_my_api_keys']['GITHUB_TOKEN'],
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
                existing_topic = st.selectbox("Select existing topic to merge with:", ['--Select a topic--'] + [t['name'] for t in st.session_state.topicsbert['topics']],index = 0, key=f"existing_topic_{radio_key}")
                with col1:
                    if st.button("Yes, merge topic", key=f"merge_{radio_key}"):
                        st.session_state['confirm_merge'] = False
                        for t in st.session_state.topicsbert['topics']:
                            if t['name'] == existing_topic:
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
                                    token=st.secrets['all_my_api_keys']['GITHUB_TOKEN'],
                                    mode = 'replace'
                                )
                                st.success(f"Topic {topic['topic']} merged successfully!")
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
                    token=st.secrets['all_my_api_keys']['GITHUB_TOKEN']
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
                token=st.secrets['all_my_api_keys']['GITHUB_TOKEN'],
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
    ASSET = 'BERTopic_results2.csv.gz'

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
    @st.cache_data(show_spinner = False, ttl = 1800)
    def get_csv_from_release(owner, repo, tag, asset, usecols = None) -> pd.DataFrame:
        token = st.secrets['all_my_api_keys']['GITHUB_TOKEN']
        headers = {"Accept": "application/vnd.github+json",
                  'Authorization': f'token {token}'}
        rel = requests.get(f'https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}', headers = headers, timeout = 60)
        rel.raise_for_status()
        rel_json = rel.json()
        asset = next((a for a in rel_json.get('assets', []) if a.get('name') == asset), None)
        if not asset:
            raise RuntimeError(f"Asset '{asset}' not found in release '{tag}'")
        url = asset['browser_download_url']
        r = requests.get(url, headers = headers, timeout = 60)
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content), compression="gzip", low_memory = False, dtype=str, usecols = usecols)
     
    required_keys = {'Title', 'Content'}
    if 'articles' not in st.session_state:
        usecols = ['Title', 'Content', 'Link', 'Published', 'University Label', '_RiskList', 'Reviewed', 'Recency', 'Source_Accuracy',
                  'Impact_Score', 'Acceleration_value', 'Location', 'Industry_Risk', 'Frequency_Score', 'Risk_Score', 'Topic', 'Probability']
        results_df = get_csv_from_release(OWNER, REPO, TAG, ASSET, usecols = usecols)
        numeric_cols = ['Recency', 'Source_Accuracy', 'Impact_Score', 'Acceleration_value', 'Location', 'Industry_Risk', 'Frequency_Score', 'Risk_Score', 'Probability']
        use_changes = Path('Model_training/BERTopic_changes.csv').is_file() and Path('Model_training/BERTopic_changes.csv').stat().st_size > 0
        changes_df = None

        if use_changes:
            try:
                changes_df = pd.read_csv('Model_training/BERTopic_changes.csv')
                def norm(s: pd.Series) -> pd.Series:
                    return s.astype(str).str.replace(r's+', ' ', regex = True).str.strip()
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
            base = results_df.drop_duplicates(subset = join_keys, keep = 'first')
            merged_df = base.merge(review_map, on = join_keys, how = 'left')
            merged_df['Reviewed'] = merged_df['Reviewed'].fillna(0).astype(int)
            st.session_state.articles = merged_df
        else:
            tmp = results_df.copy()
            tmp['Reviewed'] = 0
            tmp['Reviewed_at'] = pd.NaT
            tmp['Last_changed_at'] = pd.NaT
            st.session_state.articles = tmp

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
    for c in numeric_cols:
        if c in st.session_state.articles.columns:
            st.session_state.articles[c] = pd.to_numeric(st.session_state.articles[c], errors = 'coerce')

    ##adding to push changes to the Github repo
    def push_file_to_github(local_path:str, repo:str, dest_path:str, branch:str = "main", token:str|None = None):
        token = st.secrets['all_my_api_keys']['GITHUB_TOKEN']

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
    base_df = st.session_state.articles
    #articles = articles[articles['Published']> start_date.strftime('%Y-%m-%d')]
    #articles = articles[articles['Published']< end_date.strftime('%Y-%m-%d')]
    filtered_df = base_df[base_df['University Label'] == 1]
    filtered_df = filtered_df.drop_duplicates(subset=['Title'])
    filtered_df = filtered_df[~(filtered_df['_RiskList'] == 'No Risk')]
    if status_choice == 'Unreviewed only':
        filtered_df = filtered_df[filtered_df['Reviewed'] != 1]
    elif status_choice == 'Reviewed only':
        keys = ['Link'] if ('Link' in base_df.columns and 'Link' in st.session_state.change_log.columns) else ['Title']
        ch = st.session_state.change_log.copy()
    
        # ensure types
        ch['Reviewed'] = pd.to_numeric(ch.get('Reviewed', 0), errors='coerce').fillna(0).astype(int)
        if 'Changed_at' in ch.columns:
            ch['Changed_at'] = pd.to_datetime(ch['Changed_at'], errors='coerce')
    
        # keep last action per key, then only those with Reviewed==1
        last = (ch.sort_values('Changed_at').drop_duplicates(keys, keep='last'))
        last = last[last['Reviewed'] == 1]
    
        # merge onto articles so schema/index are consistent for rendering
        keep_cols = [c for c in ['Reviewed','Reviewed_at','Changed_at'] if c in last.columns]
        filtered_df = base_df.merge(last[keys + keep_cols], on=keys, how='inner', suffixes = ('', '_chg'))
        if 'Reviewed_chg' in filtered_df.columns:
            filtered_df['Reviewed'] = filtered_df['Reviewed_chg'].fillna(filtered_df.get('Reviewed', 0)).astype(int)
            filtered_df.drop(columns = [c for c in ['Reviewed_chg'] if c in filtered_df.columns], inplace = True)

    start_date = pd.to_datetime(start_date).tz_localize(ZoneInfo("America/Chicago")).tz_convert('UTC')
    end_date = (pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)).tz_localize(ZoneInfo("America/Chicago")).tz_convert('UTC')
    filtered_df['Published'] = pd.to_datetime(filtered_df['Published'], errors = 'coerce', utc = True)
    filtered_df = filtered_df[filtered_df['Published'].between(start_date, end_date, inclusive = 'both')]
    filtered_df = filtered_df.sort_values('Published', ascending = False, na_position = 'last')

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

    PAGE_SIZE = st.sidebar.selectbox('Items per Page', [10, 20, 30, 50], index =1)
    total = len(filtered_df)
    max_page = max(1, (total + PAGE_SIZE - 1)//PAGE_SIZE)

    if 'page_num' not in st.session_state:
        st.session_state.page_num = 1
    st.session_state.page_num = st.sidebar.number_input(
        'Page', min_value = 1, max_value = max_page, value = st.session_state.page_num, step =1
    )

    start = (st.session_state.page_num - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    st.caption(f"Showing {start + 1} to {min(end, total)} of {total} articles")
    page_df = filtered_df.iloc[start:end]

    with open('Model_training/topics_BERT.json', 'r', encoding = 'utf-8') as f:
        name_map = {int(t['topic']): t['name'] for t in json.load(f)['topics']}

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

    for _, article in page_df.iterrows():
        reviewed = bool(int(article.get('Reviewed', 0)))
        badge = "✅ Reviewed" if reviewed else "Not reviewed"
        title = str(article.get("Title", ""))[:100]
    
        
        raw = article.get("_RiskList", "[]")
        if isinstance(raw, list):
            predicted = raw
        elif isinstance(raw, str):
            s = raw.strip()
            if s.lower() in ("", "none", "no risk"):
                predicted = ["No Risk"]
            else:
                parts = [r.strip() for r in s.split(';') if r.strip()]
                predicted = parts if parts else ["No Risk"]   # keep all phrases if you ever have "a; b"
        else:
            predicted = ["No Risk"]

        if not match_any(predicted, filtered_risks):
            continue

        title = str(article.get("Title", ""))[:100]
        
        if title:
           
            tid = coerce_topic_scalar(article.get('Topic'))
            article['Topic'] = tid
            
            article['Topic_name'] = name_map.get(tid, 'Unlabeled Topic')
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
                    st.markdown("**Topic:** ")
                    st.markdown(article['Topic_name'])
    
                # --- Quick review toggle ---
                c1, c2 = st.columns(2)
                with c1:
                    if not reviewed:
                        if st.button("Mark as reviewed", key=f"mark_{article.get('Link')}"):
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
                        if st.button("Unmark reviewed", key=f"unmark_{article}"):
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
                    if st.button('Hide this topic', key = f'hide_topic_{tid}_{article}'):
                        if tid != -1:
                            hidden_topic_ids.add(int(tid))
                            save_hidden_topics(hidden_topic_ids)
                            st.success(f"Hid topic {tid} - {article['Topic_name']}")
                            st.rerun()

                matched_risks = [
                    opt for opt in all_possible_risks
                    if any(opt.lower() == str(p).lower() for p in predicted if isinstance(p, str))
                ]
                
                st.markdown("**Predicted Risks:** " + (", ".join(matched_risks) if matched_risks else "No Risk"))
                
                
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

                    with tab2:
                        options = [0.0, 1.0,2.0,3.0,4.0,5.0]
                        with st.form(f"manual_edit_form_{article}"):
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
                                default_index = pairs.index(default_pair)
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
                                    key = f"edit_c_{article}"
                                )
                                selected_risks = [choice[1]]
                            col1, col2, col3, col4, col5, col6, col7 =  st.columns(7)
                            with col1:
                                upd_recency_value = st.number_input('Recency Risk', min_value = 0.0, max_value = 5.0, step = 1.0, value= float(article['Recency_Upd'] if pd.notna(article['Recency_Upd']) else article['Recency']), key =f"recency_input_{article}")
                            with col2:
                                upd_acceleration_value = st.number_input('Acceleration Risk',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(article['Acceleration_value_Upd'] if pd.notna(article['Acceleration_value_Upd']) else article['Acceleration_value']),key =f"acceleration_input_{article}")
                            with col3:
                                upd_source_accuracy =st.number_input('Source Accuracy',  min_value=0.0, max_value = 5.0, step = 1.0, value= float(article['Source_Accuracy_Upd'] if pd.notna(article['Source_Accuracy_Upd']) else article['Source_Accuracy']),key =f"source_input_{article}")
                            with col4:
                                upd_impact_score = st.number_input('Impact Score',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(article['Impact_Score_Upd'] if pd.notna(article['Impact_Score_Upd']) else article['Impact_Score']),key =f"impact_input_{article}")
                            with col5:
                                upd_location=st.number_input('Location Risk',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(article['Location_Upd'] if pd.notna(article['Location_Upd']) else article['Location']),key =f"location_input_{article}")
                            with col6:
                                upd_industry_risk = st.number_input('Industry Risk',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(article['Industry_Risk_Upd'] if pd.notna(article['Industry_Risk_Upd']) else article['Industry_Risk']),key =f"industry_input_{article}")
                            with col7:
                                upd_frequency_score = st.number_input('Frequency Score', min_value=0.0, max_value = 5.0, step = 1.0, value=float(article['Frequency_Score_Upd'] if pd.notna(article['Frequency_Score_Upd']) else article['Frequency_Score']),key =f"frequency_input_{article}")

                            st.markdown('Please provide a reason for the changes made to the risk labels:')
                            reason = st.text_area("Reason for changes", placeholder="Explain the changes made to the risk labels.", key=f"reason_{article}")
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
