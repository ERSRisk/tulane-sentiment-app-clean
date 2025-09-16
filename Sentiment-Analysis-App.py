import streamlit as st
import requests
from google import genai
import pandas as pd
import re
from datetime import timedelta
from streamlit_tags import st_tags
import plotly.express as px
from newspaper import Article
from google.genai.errors import ClientError
import random
import time
import asyncio
from lxml.html.clean import Cleaner
from dateutil import parser
import datetime
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



st.set_page_config(page_title="Tulane Risk Dashboard")
st.sidebar.title("Navigation")
st.sidebar.markdown("Select a tool:")
selection = st.sidebar.selectbox("Choose a tool:", ["News Sentiment", "X Sentiment", "Article Risk Review", "Unmatched Topic Analysis", "Risk/Event Detector"])

if "current_tab" not in st.session_state:
    st.session_state.current_tab = selection

# If switching tabs, clear session except the current tab
if st.session_state.current_tab != selection:
    keys_to_keep = {"current_tab"}
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    st.session_state.current_tab = selection

if selection == "News Sentiment":
    # Setting up the APIs for News API and Gemini API
    NEWS_API_KEY = st.secrets["all_my_api_keys"]["NEWS_API_KEY"]
    GEMINI_API_KEY = st.secrets["all_my_api_keys"]["GEMINI_API_KEY_X"]






    # Configure Gemini API
    client = genai.Client(api_key=GEMINI_API_KEY)


    # Streamlit UI
    st.title("Tulane University: Sentiment Analysis from News")




    # Make it so someone can type in their own keywords to customize the search
    search = st_tags(label="Enter your values (press Enter to separate keywords):",
                    text="Add a new value...",
                    value=["Tulane"],  # Default values
                    suggestions=["Tulane University"],  # Optional suggestions
                    key="1")

    # Date range selection
    start_date = st.date_input("Start Date", value= datetime.date.today() - timedelta(days = 7))
    end_date = st.date_input("End Date", value=datetime.date.today())
    timezone_option = st.selectbox(
        "Select Timezone for Article Timestamps:",
        options=["UTC", "CST", "CDT"],
        index=2  # Default to CDT
    )

    if search:
        st.session_state.search_ran = True

    ## Checkbox for including sports news. Selecting the checkbox will include sports news in the search.
    sports = st.checkbox("Include sports news")


    ## Checkbox for using cache. Unchecking this will allow for debugging.
    use_cache = st.checkbox("Use cache (uncheck for debugging purposes)", value=True)

    if st.session_state.get("search_ran"):
        ## This first function fetches the news articles from the News API based on the search keywords and date range.
        @st.cache_data(show_spinner=False, persist=True)
        def fetch_news(search, start_date, end_date, sports):
            if sports:
                news_url = (
                    f"https://newsapi.org/v2/everything?q={search}&"
                    f"from={start_date}&to={end_date}&sortBy=popularity&apiKey={NEWS_API_KEY}"
                )
            else:
                news_url = (
                    f"https://newsapi.org/v2/everything?q={search} NOT sports NOT Football NOT basketball&"
                    f"from={start_date}&to={end_date}&sortBy=popularity&apiKey={NEWS_API_KEY}"
                )
            response = requests.get(news_url)
            if response.status_code == 200:
                news_data = response.json()
                return news_data.get("articles", [])  # Return the articles
            else:
                return []


        ## This function fetches the full content of an article using the Newspaper3k library.
        ## The News API only provides a snippet of the article, so this function is used to get the full text.
        def fetch_content(url):
            try:
                article = Article(url)
                article.download()
                article.parse()
                return article.text
            except Exception as e:
                return None


        # This function combines the truncated content from the News API with the full content fetched from the article URL.
        # It also extracts the date and time from the article's publishedAt field and formats it according to the selected timezone.
        @st.cache_data(show_spinner=False, persist=True)
        def get_articles_with_full_content(articles, timezone="CDT"):
            """Replace truncated content with full article text and extract formatted date and time"""
            updated_articles = []
            seen_titles = set()

            #Determine offset based on selected timezone
            if timezone == "UTC":
                offset = timedelta(hours=0)
                tz_label = "UTC"
            elif timezone == "CST":
                offset = timedelta(hours=-6)
                tz_label = "CST"
            elif timezone == "CDT":
                offset = timedelta(hours=-5)
                tz_label = "CDT"
            else:
                offset = timedelta(hours=0)
                tz_label = "UTC"




            for article in articles:
                title = article["title"]
                if title in seen_titles:
                    continue  # Skip duplicate
                seen_titles.add(title)




                #Get full text if the content is truncated
                full_text = fetch_content(article['url']) or article.get('content')


                #Parse publishedAt and split into date and time
                original_dt_str = article.get("publishedAt", "N/A")




                #try to parse and convert
                try:
                    original_dt = parser.parse(original_dt_str)
                    adjusted_dt = original_dt + offset  # Convert from UTC to CST
                    adjusted_date = adjusted_dt.strftime("%m/%d/%Y")
                    adjusted_time = adjusted_dt.strftime("%I:%M %p ") + tz_label
                except Exception:
                    adjusted_date = "N/A"
                    adjusted_time = "N/A"




                updated_articles.append({
                    "title": article["title"],
                    "description": article.get("description", "No description available."),
                    "content": full_text if full_text else article["content"],
                    "url": article["url"],
                    "original_datetime": original_dt_str,
                    "adjusted_date": adjusted_date,
                    "adjusted_time": adjusted_time
                })
            return updated_articles



        # This function formats the articles into a string that can be sent to the Gemini API for sentiment analysis.
        # It includes the title, description, content, and URL of each article.
        @st.cache_data(show_spinner=False, persist=True)
        def format_articles_for_prompt(articles):
            """Format the articles in a way that can be sent to Gemini."""
            return "\n\n".join(
                [f"Title: {article['title']}\nDescription: {article['description']}\nContent: {article['content']}\nURL: {article['url']}"
                for article in articles])


        # This function analyzes the sentiment of the articles using the Gemini API.
        # It sends a prompt to the API and returns the response.
        # The prompt includes instructions for the API to analyze the sentiment based on the keywords provided.
        @st.cache_data(show_spinner=False, persist=True)
        def analyze_sentiment(text_to_analyze, search, sports, retries=5):
            for attempt in range(retries):
                try:
                    if sports:
                        sentiment_prompt = (
                            "Analyze the sentiment of the following news articles in relation to the keywords: "
                            f"'{search}'.\n"
                            "Assume all articles affect Tulane's reputation positively, neutrally, or negatively. \n"
                            "Then, consider how the keywords also get discussed or portrayed in the article.\n"
                            "Provide an overall sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, and 1 is very positive(This is a continuous range)) \n"
                            "Provide a summary of the sentiment and key reasons why the sentiment is positive, neutral, or negative, "
                            "specifically in relation to the keywords.\n"
                            "Make sure that you include the score from -1 to 1 in a continuous range (with decimal places) and include the title, "
                            "sentiment score, summary, and a statement explaining how the article relates to the keywords.\n"
                            "Separate article info by double newlines and always include 'Title:' before the headline and 'Sentiment:' before the score.\n"
                            "Only judge the sentiment for each article in terms of how it mentions the keywords. Max amount of titles should be 100.\n\n"
                            "If an article title has already been seen before, do not analyze it again.\n"
                            "If Tulane is mentioned anywhere in the article — even in passing or in an author affiliation — state that clearly in your summary.\n"
                            "Tulane was found in the text. Here is the full content for analysis.\n"
                            f"{text_to_analyze}"
                        )
                    else:
                        sentiment_prompt = (
                            "Analyze the sentiment of the following news articles in relation to the keywords: "
                            f"'{search}'.\n"
                            "Assume all articles affect Tulane's reputation positively, neutrally, or negatively. \n"
                            "Then, consider how the keywords also get discussed or portrayed in the article.\n"
                            "Provide an overall sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, and 1 is very positive(This is a continuous range)) \n"
                            "Provide a summary of the sentiment and key reasons why the sentiment is positive, neutral, or negative, "
                            "specifically in relation to the keywords.\n"
                            "Make sure that you include the score from -1 to 1 in a continuous range (with decimal places) and include the title, "
                            "sentiment score, summary, and a statement explaining how the article relates to the keywords.\n"
                            "Separate article info by double newlines and always include 'Title:' before the headline and 'Sentiment:' before the score.\n"
                            "If you encounter any articles related to sports, please exclude them from the analysis. Sports articles do not need to be summarized. \n"
                            "Only judge the sentiment for each article in terms of how it mentions the keywords. Max amount of titles should be 100.\n\n"
                            "If an article title has already been seen before, do not analyze it again.\n"
                            "If Tulane is mentioned anywhere in the article — even in passing or in an author affiliation — state that clearly in your summary.\n"
                            "Tulane was found in the text. Here is the full content for analysis.\n"
                            f"{text_to_analyze}"
                        )
                    gemini_response = client.models.generate_content(model="gemini-1.5-flash", contents=[sentiment_prompt])
                    return gemini_response.text if gemini_response and gemini_response.text else ""
                except ClientError as e:
                    if "RESOURCE_EXHAUSTED" in str(e):
                        wait_time = 60  # Default wait time (1 minute)
                        retry_delay_match = re.search(r"'retryDelay': '(\d+)s'", str(e))
                        if retry_delay_match:
                            wait_time = int(retry_delay_match.group(1))  # Use API's recommended delay

                        print(f"⚠️ API quota exceeded. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ API request failed: {e}")
                        return "❌ API error encountered."
                except requests.exceptions.ConnectionError:
                    wait_time = 2 ** attempt + random.uniform(0, 1)
                    print(f"Connection error. Waiting for {wait_time:.2f} seconds before retrying...")
                    time.sleep(wait_time)
                return "❌ API failed after multiple attempts."




        # This function processes a batch of articles concurrently using asyncio.
        semaphore = asyncio.Semaphore(5)


        # This function limits the number of concurrent tasks to avoid overwhelming the API.
        async def limited_process(batch_df, search, batch_size, total_batches, timezone):
            async with semaphore:
                return await asyncio.to_thread(process_batch, batch_df, search, batch_size, total_batches, timezone)




        # This function processes the articles in batches and returns the responses.
        # It divides the articles into smaller batches and processes them concurrently.
        # It also handles the pagination of the articles by keeping track of the batch number and total batches.
        async def analyze_in_batches_concurrent(articles, search, sports, timezone, batch_size=10):
            all_responses = []
            total_batches = len(articles) // batch_size + (1 if len(articles) % batch_size != 0 else 0)
            print(f"Total batches: {total_batches}")
            batch_tasks = []
            for i in range(0, len(articles), batch_size):
                batch_df = articles[i:i + batch_size]




                task = limited_process(batch_df, search, i // batch_size + 1, total_batches, timezone)
                batch_tasks.append(task)

            all_responses = await asyncio.gather(*batch_tasks)
            return '\n\n'.join(all_responses)




        # This function runs the asynchronous batch processing and returns the final responses.
        # It uses asyncio.run to execute the asynchronous function and waits for it to complete.
        # It also handles the pagination of the articles by keeping track of the batch number and total batches.
        def run_async_batches(articles, search, sports, timezone, batch_size = 10):
            return asyncio.run(analyze_in_batches_concurrent(articles, search, sports, timezone, batch_size))


        # This function caches the responses from the Gemini API to avoid redundant API calls.
        # It uses Streamlit's caching mechanism to store the responses based on the input parameters.
        @st.cache_data(show_spinner=False, persist=True)
        def cached_gemini_response(articles, search, sports, timezone):
            return run_async_batches(articles, search, sports, timezone, batch_size=10)


        # This function processes a batch of articles and returns the response from the Gemini API.
        # It formats the articles for the prompt and sends them to the API for sentiment analysis.
        # It also handles the pagination of the articles by keeping track of the batch number and total batches.
        def process_batch(batch_df, search, i, total_batches, timezone):


            processed_articles = get_articles_with_full_content(batch_df, timezone=timezone)
            formatted_batch = format_articles_for_prompt(processed_articles)
            print(f"Processing batch {i} of {total_batches}...")
            response = analyze_sentiment(formatted_batch, search, sports)
            return response


        if "slider_value" not in st.session_state:
            st.session_state.slider_value = (-1.0, 1.0)


        # This function handles the display of the slider for filtering the sentiment scores.
        if st.button('Search') or "slider_shown" in st.session_state:
            search = '+'.join(search)
            if use_cache:
                articles = fetch_news(search, start_date, end_date, sports)
                articles = get_articles_with_full_content(articles, timezone=timezone_option)
                unique_articles = []
                seen_titles = set()
                for article in articles:
                    if article['title'] not in seen_titles:
                        unique_articles.append(article)
                        seen_titles.add(article['title'])
                articles = unique_articles
            else:
                fetch_news.clear()
                articles = fetch_news(search, start_date, end_date, sports)
                articles = get_articles_with_full_content(articles, timezone=timezone_option)
                unique_articles = []
                seen_titles = set()


                for article in articles:
                    if article['title'] not in seen_titles:
                        unique_articles.append(article)
                        seen_titles.add(article['title'])
                articles = unique_articles

            if not articles:
                st.write("No articles found.")
            else:
                if use_cache:
                    gemini_response_text = cached_gemini_response(articles, search, sports, timezone_option)
                else:
                    gemini_response_text = run_async_batches(articles, search, sports, timezone_option, batch_size=10)

                # This function processes the response from the Gemini API and formats it into a DataFrame.
                # It extracts the title, sentiment score, summary, and other relevant information from the response.
                def text_to_dataframe(text, articles):
                    rows = []
                    sections = re.split(r'Title:\s*', text)[1:]




                    for section in sections:
                        title_match = re.match(r'(.*?)\nSentiment:', section, re.DOTALL)
                        sentiment_match = re.search(r'Sentiment:\s*(-?\d+\.?\d*)', section)




                        if title_match and sentiment_match:
                            title = title_match.group(1).strip()
                            sentiment = float(sentiment_match.group(1))
                            article_data = next((article for article in articles if article['title'].strip().lower() == title.lower()), None)
                            if article_data is not None:
                                rows.append({
                                'Title': title,
                                'Sentiment': sentiment,
                                'URL': article_data.get('url'),
                                'Original Datetime': article_data.get('original_datetime', 'N/A'),
                                'Adjusted Date': article_data.get('adjusted_date', 'N/A'),
                                'Adjusted Time': article_data.get('adjusted_time', 'N/A'),
                                'Full Article Text':article_data.get('content', 'N/A')
                                })
                            else:
                                rows.append({
                                'Title': title,
                                'Sentiment': sentiment,
                                'URL': 'Not Found',
                                'Original Datetime': 'Not Found',
                                'Adjusted Date': 'Not Found',
                                'Adjusted Time': 'Not Found'
                                })





                    return pd.DataFrame(rows)












                df = text_to_dataframe(gemini_response_text, articles)
                sentiment_counts = df['Sentiment'].value_counts()

                st.header("Sentiment Score Summary")
                st.write("")
                # Plot sentiment score summary
                st.bar_chart(sentiment_counts)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Sentiment Score", round(df['Sentiment'].mean(), 2))
                with col2:
                    st.metric("Number of News Stories", len(df))


        # This function checks the overall sentiment of the articles and displays a message accordingly.
                if df['Sentiment'].mean() >= 0.1:
                    st.write("Overall sentiment is positive.")
                elif df['Sentiment'].mean() <= -0.1:
                    st.write("Overall sentiment is negative.")
                else:
                    st.write("Overall sentiment is neutral.")

                st.write("---")
                st.header("News Stories")


                st.session_state.slider_shown = True
                st.session_state.slider_value = st.slider("Sentiment Filter", -1.0, 1.0, (-1.0, 1.0), 0.1,)

                st.write("")
                filtered_df = df[(df['Sentiment'] >= st.session_state.slider_value[0]) &
                        (df['Sentiment'] <= st.session_state.slider_value[1])]


        # This function displays the filtered articles based on the sentiment score.
                for _, row in filtered_df.iterrows():
                    st.markdown("---") #before each row
                    st.markdown(f"###  **[{row['Title']}]({row['URL']})**")
                    st.markdown(f"**Date & Time:** {row.get('Adjusted Date', 'N/A')} at {row.get('Adjusted Time', 'N/A')}")
                    st.markdown(f"🔹 **Sentiment Score:** `{row['Sentiment']}`")

                    # Grab summary from the original text using the title
                    pattern = rf"Title:\s*{re.escape(row['Title'])}\s*.*?Sentiment:\s*-?\d+\.?\d*\s*Summary:\s*(.*?)(?:\n|$)"
                    match = re.search(pattern, gemini_response_text, re.DOTALL)
                    if match:
                        summary = match.group(1).strip()
                        st.markdown(f"**Summary:** {summary}")
                    else:
                        st.markdown("⚠️ Summary not found.")
                    st.write("---")

                #format the date range into a MM-DD-YYYY format
                start_str = start_date.strftime("%m-%d-%Y")
                end_str = end_date.strftime("%m-%d-%Y")




                #create a dynamic name
                df_name = f"TU_News_{start_str}_to_{end_str}"




                #display the Dataframe with the dynamic name
                st.header(f"Dataframe of Results")
                st.dataframe(df)




                #download button for the dataframe
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Dataframe",
                    data=csv,
                    file_name=f"{df_name}.csv",
                    mime="text/csv",
                )




        st.markdown("---")
    #st.markdown("🔍 Built by Luke Roosa, Samuel Centeno, & Illia Kunin | Powered by NewsAPI & Gemini")
if selection == "X Sentiment":
    nest_asyncio.apply()
    #This function is used to analyze the tweets using Google Gemini API
    # It takes a tweet as input and returns a JSON response with a description and whether the tweet is related to sports or not.

    #Define the API keys for X and Gemini
    GEMINI_API_KEY_X = st.secrets["all_my_api_keys"]["GEMINI_API_KEY_X"]
    X_API_KEY = st.secrets["all_my_api_keys"]["X_API_KEY"]


    #adding this option to run batches
    semaphore = asyncio.Semaphore(3)

    async def limited_process(batch_df, search, batch_size, total_batches):
            async with semaphore:
                return await asyncio.to_thread(process_batch, batch_df, search, batch_size, total_batches)

    async def analyze_in_batches_concurrent_X(tweets, search, sports, batch_size=10):
            all_responses = []
            total_batches = len(tweets) // batch_size + (1 if len(tweets) % batch_size != 0 else 0)
            print(f"Total batches: {total_batches}")
            batch_tasks = []
            for i in range(0, len(tweets), batch_size):
                batch_df = tweets[i:i + batch_size]




                task = limited_process(batch_df, search, i // batch_size + 1, total_batches)
                batch_tasks.append(task)

            all_responses = await asyncio.gather(*batch_tasks)
            return pd.concat(all_responses)

    async def run_async_batches_X(tweets, search, sports, batch_size = 10):
        return asyncio.run(analyze_in_batches_concurrent_X(tweets, search, sports, batch_size = 10))

    def process_batch(batch_df, search, i, total_batches):
        print(f"Processing batch {i} of {total_batches}...")
        formatted_tweets = "\n\n".join([f"{tweet.text}" for tweet in batch_df])
        analysis_list = analyze_sentiment_X(formatted_tweets, search, sports)
        response = []
        if not analysis_list:
            print("analysis list is missing")
        elif len(analysis_list) != len(batch_df):
            print(f"Got {len(analysis_list)} responses for {len(batch_df)} tweets.")
        for tweet, analysis in zip(batch_df, analysis_list):
            result = {
            "created_at": tweet.created_at,
            "text": tweet.text,
            "link": f"https://twitter.com/{tweet.author_id}/status/{tweet.id}",
            "description": analysis.get("description", "parse error"),
            "sentiment": analysis.get("sentiment", 0),
            "summary": analysis.get("summary", "parse error"),
            "is_sport": analysis.get("is_sport", 0),
            "affiliation": analysis.get("affiliation", 0)
            }
            if result['is_sport'] == 0 and result['affiliation'] == 1:
                response.append(result)
        return pd.DataFrame(response)



    #This function fetches tweets from Twitter API based on the search term and date range provided by the user.
    # It uses the Tweepy library to interact with the Twitter API and returns a DataFrame with the tweet data.
    # The function takes the following parameters:
    # search: The search term to look for in tweets.
    # start_date: The start date for the tweet search.
    # end_date: The end date for the tweet search.
    # no_of_tweets: The number of tweets to fetch.
    def fetch_twits(search, start_date, end_date, no_of_tweets):
        import datetime
        client = tweepy.Client(bearer_token=X_API_KEY)
        response = client.search_recent_tweets(
            query=search,
            max_results=100,
            tweet_fields=["created_at"],
            start_time=start_date.isoformat() + "Z",
            end_time=(datetime.datetime.combine(end_date, datetime.time.min)).isoformat() + "Z"
        )
        tweets = response.data
        if not tweets:
            st.warning('No tweets were extracted. Check X source')
            st.session_state.x_search_ran = False
            st.stop()
        return tweets

    def analyze_sentiment_X(formatted_tweet_block, search, sports, retries=5):
        flagged_keywords = ["1. Civil Rights", "2. Antisemitism", "3. Federal Grants", "4. Contracts", 
                                "5. Discrimination", "6. Education Secretary", "7. Investigation", "8. Lawsuit", 
                                "9. Executive Order", "10. Title IX", "11. Transgender Athletes", "12. Diversity, Equity, and Inclusion (DEI)", 
                                "13. Funding Freeze, funding frost", "14. University Policies", "15. Student Success", '16. Allegations', "17. Compliance", 
                                "18. Oversight", "19. Political Activity", 
                                "20. Community Relations"]
        client = genai.Client(api_key=GEMINI_API_KEY_X)
        prompt = f"""
                Analyze the following tweets. For each one, return a JSON object with:
                - "description": a short description of the tweet,
                - "sentiment": a sentiment score from -1 to 1 (where -1 is very negative, 0 is neutral, and 1 is very positive) based on its relation with the keywords '{search}' and {flagged_keywords}. If it's an academic study conducted by Tulane that is recognized, give it a high score. Very low scores should be given to topics regarding the following issues if shed in a negative light: {flagged_keywords},
                - "summary": a summary of the sentiment and key reasons why the sentiment is positive, neutral, or negative based on its relation with the keywords '{search}',
                - "is_sport": 1 if the tweet is related to sports, if the word 'player', 'playoffs', 'NFL', 'season', or any other sport related is found in the text of the post, or the description contains references to sports seasons, match results, baseball, football, player recruitment; else, give it a 0.
                - "affiliation": 1 if the tweet is affiliated with Tulane directly and has bearing on Tulane's reputation as an organization; else, if the post is indirectly related to Tulane or has no bearing on the organization, give it a 0.

                Respond as a JSON array of objects, one per tweet, in the order presented.

                Tweets:
                {formatted_tweet_block}
                """
        models = ["gemini-1.5-pro", "gemini-2.5-pro-preview-05-06"]
        for attempt in range(retries): 
            for model in models:
                try:

                    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                    if response.candidates and response.candidates[0].content.parts:
                        raw_text = response.candidates[0].content.parts[0].text
                        print(raw_text)
                        cleaned_text = re.sub(r"```json|```|\n|\s{2,}", "", raw_text).strip()
                        return json.loads(cleaned_text)

                except ClientError as e:
                    if "RESOURCE_EXHAUSTED" in str(e):
                        wait_time = 60
                        retry_delay_match = re.search(r"'retryDelay': '(\d+)s'", str(e))
                        if retry_delay_match:
                            wait_time = int(retry_delay_match.group(1))
                        print(f"⚠️ API quota exceeded. Retrying in {wait_time} seconds...")
                        time.sleep(60)
                except Exception:
                    continue
        return print("API Failed. Check API Key or model.")


    #Setting the page title and layout for the UI
    st.title("Tulane University: Sentiment Analysis from X")
    search = st_tags(
        label="Enter your values (press Enter to separate keywords):",
        text="Add a new value...",
        value=["Tulane"],  # Default values
        suggestions=["Tulane University"],  # Optional suggestions
        key="1"
    )
    #Adding a date input for the user to select the start and end dates for the tweet search
    start_date = st.date_input("Start Date", value= datetime.date.today() - datetime.timedelta(days = 6))
    start_date= datetime.datetime.combine(start_date, datetime.time(0, 0)) + datetime.timedelta(hours=1)
    end_date = st.date_input("End Date", value=datetime.date.today())
    search_button = st.button("Search")

    sports= st.checkbox("Include sports news")
    pass
    pass


    # Store search trigger persistently
    # Track button click
    if search_button:
        st.session_state.x_search_ran = True
        st.session_state.x_results_ready = False  # Reset
    # Fetch only when search button is pressed
    if st.session_state.get("x_search_ran", False):
        tweets = fetch_twits(search, start_date, end_date, 100)
        with open("tweets.json", "w") as f:
            json.dump([tweet.data for tweet in tweets], f)
        print('Fetched tweets:', tweets[:2] if tweets else "None")
        if not tweets:
            st.warning('No posts were extracted. Check X source.')
            st.session_state.x_search_ran = False
            st.stop()    
        df = asyncio.get_event_loop().run_until_complete(run_async_batches_X(tweets, search, sports, batch_size=10))
        st.session_state.x_df = df
        st.session_state.x_search_ran = False
        st.session_state.x_results_ready = True

    # === Display results if ready ===
    if st.session_state.get("x_results_ready", False):
        df = st.session_state.get("x_df", None)

        if df is None or df.empty:
            st.warning("No tweets found for the given search term and date range.")
            st.stop()

        slider_value = st.slider(
            "Sentiment Filter",
            min_value=-1.0,
            max_value=1.0,
            value=(-1.0, 1.0),
            step=0.1,
            key="slider_value"
        )

        # Filter tweets based on sentiment
        df_filtered = df[
            (df['sentiment'] >= slider_value[0]) &
            (df['sentiment'] <= slider_value[1])
        ]

        if df_filtered.empty:
            st.warning("No tweets found in this sentiment range.")
            st.stop()

        # Clean usernames for display
        df_filtered['text'] = df_filtered['text'].apply(lambda x: re.sub(r"@\w+", "@user", x))

        # Chart
        st.subheader("Sentiment Distribution")
        sentiment_counts = df_filtered['sentiment'].value_counts().sort_index()
        st.bar_chart(sentiment_counts)

        # Display tweets
        for _, row in df_filtered.iterrows():
            st.markdown(f"**Created At:** {row['created_at'].strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Link:** [Tweet Link]({row['link']})")
            st.markdown(f"**Text:** {row['text']}")
            st.markdown(f"**Description:** {row['description']}")
            st.markdown(f"**Sentiment:** {row['sentiment']}")
            st.markdown(f"**Summary:** {row['summary']}")
            st.write("---")

        # Full cleaned table
        st.write(df_filtered.drop(columns=["is_sport"]))
if selection == "Unmatched Topic Analysis":
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
    if 'unmatched' not in st.session_state:
        if os.path.exists('Model_training/unmatched_topics.json'):
            with open('Model_training/unmatched_topics.json', 'r') as f:
                st.session_state.unmatched = json.load(f)
        else:
            st.session_state.unmatched = []

    if 'topicsbert' not in st.session_state:
        if os.path.exists('Model_training/topics_BERT.json'):
            with open('Model_training/topics_BERT.json', 'r') as f:
                st.session_state.topicsbert = json.load(f)
        else:
            st.session_state.topicsbert = []
    
    try:
        if 'discarded' not in st.session_state:
            if os.path.exists('Model_training/discarded_topics.json'):
                with open('Online_Extraction/discarded_topics.json', 'r') as f:
                    discarded_topics = json.load(f)
                if not isinstance(discarded_topics, list):
                    discarded_topics = [discarded_topics]
                st.session_state.discarded = discarded_topics
    except Exception:
        discarded_topics = []
        st.session_state.discarded = []
    st.title('Unmatched Topics Analysis')

    for topic in st.session_state.unmatched:
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
                        new_topic = {
                            'topic': topic['topic'],
                            'name': topic['name'],
                            'keywords': topic['keywords'],
                            'documents': topic['documents']
                        }
                        st.session_state.topicsbert.append(new_topic)
                        resp = push_file_to_github('Model_training/topics_bert.json', repo = 'ERSRisk/tulane-sentiment-app-clean',
                                                              dest_path = 'Model_training/topics_bert.json', branch = 'main')
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
                with col1:
                    if st.button("Yes, merge topic", key=f"merge_{radio_key}"):
                        st.session_state['confirm_merge'] = False
                        existing_topic = st.selectbox("Select existing topic to merge with:", ['--Select a topic--'] + [t['name'] for t in st.session_state.topicsbert],index = 0, key=f"existing_topic_{radio_key}")
                        for t in st.session_state.topicsbert:
                            if t['name'] == existing_topic:
                                if isinstance(t['documents'], str):
                                    t['documents'] = [t['documents']]
                                t['documents'].extend(topic['documents'])

                            # Ensure keywords are lists
                                if isinstance(t['keywords'], str):
                                    t['keywords'] = [k.strip() for k in t['keywords'].split(',')]
                                    new_keywords = [k.strip() for k in topic['keywords'].split(',')] if isinstance(topic['keywords'], str) else topic['keywords']
                                t['keywords'].extend(new_keywords)
                                resp1 = push_file_to_github('Model_training/topics_bert.json', repo = 'ERSRisk/tulane-sentiment-app-clean',
                                                              dest_path = 'Model_training/topics_bert.json', branch = 'main')
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
            resp2 = push_file_to_github('Model_training/discarded_topics.json', repo = 'ERSRisk/tulane-sentiment-app-clean',
                                                              dest_path = 'Model_training/discarded_topics.json', branch = 'main')

            unmatched_json = [t for t in st.session_state.unmatched if t['topic'] != topic['topic']]
            resp3 = push_file_to_github('Model_training/unmatched_topics.json', repo = 'ERSRisk/tulane-sentiment-app-clean',
                                                              dest_path = 'Model_training/unmatched_topics.json', branch = 'main')

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

    def get_csv_from_release(owner, repo, tag, asset) -> pd.DataFrame:
        token = st.secrets['all_my_api_keys']['GITHUB_TOKEN']
        headers = {"Accept": "application/vnd.github+json",
                  'Authorization': f'token {token}'}
        rel = requests.get(f'https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}', headers = headers, timeout = 60)
        rel.raise_for_status()
        rel_json = rel.json()
        asset = next((a for a in rel_json.get('assets', []) if a.get('name') == asset), None)
        if not asset:
            raise RuntimeError(f"Asset '{asset_name}' not found in release '{tag}'")
        url = asset['browser_download_url']
        r = requests.get(url, headers = headers, timeout = 60)
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content), compression="gzip")
     
    required_keys = {'Title', 'Content'}
    if 'articles' not in st.session_state:
        results_df = get_csv_from_release(OWNER, REPO, TAG, ASSET)
        use_changes = Path('Model_training/BERTopic_changes.csv').is_file() and Path('Model_training/BERTopic_changes.csv').stat().st_size > 0
        changes_df = None

        if use_changes:
            try:
                changes_df = pd.read_csv('Model_training/BERTopic_changes.csv')
                if not changes_df.empty and required_keys.issubset(changes_df.columns):
                    if 'Changed_at' in changes_df.columns:
                        changes_df['Changed_at'] = pd.to_datetime(changes_df['Changed_at'], errors = 'coerce')
                    if 'Reviewed' not in changes_df.columns:
                        changes_df['Reviewed'] = 0
                    if 'Reviewed_at' not in changes_df.columns:
                        changes_df['Reviewed_at'] = pd.NaT

                    review_cols = ['Title', 'Content', 'Reviewed', 'Reviewed_at', 'Changed_at']
                    agg = {
                        'Reviewed': 'max',
                        'Reviewed_at': 'max',
                        'Changed_at': 'max'
                    }
                    review_map = (changes_df[review_cols].groupby(['Title', 'Content'], as_index = False).agg(agg)
                                 .rename(columns = {'Changed_at': 'Last_changed_at'}))
                else:
                    changes_df = None
            except Exception as e:
                changes_df = None
        if changes_df is not None:
            base = results_df.drop_duplicates(subset = ['Title', 'Content'], keep = 'first')
            merged_df = base.merge(review_map, on = ['Title', 'Content'], how = 'left')
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
    if status_choice == 'Unreviewed only':
        filtered_df = filtered_df[filtered_df['Reviewed'] != 1]
    elif status_choice == 'Reviewed only':
        filtered_df = filtered_df[filtered_df['Reviewed'] == 1]
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days = 1) - pd.Timedelta(microseconds = 1)
    filtered_df['Published'] = pd.to_datetime(filtered_df['Published'], errors = 'coerce')
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


    for idx in page_df.index:
        article= st.session_state.articles.loc[idx]
        idx = article.name
        if pd.isna(article.get('Title')) or pd.isna(article.get('Content')):
            continue
        reviewed = bool(article.get('Reviewed', 0))
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
                st.metric('Risk Score', article['Risk_Score_y'])
    
                # --- Quick review toggle ---
                c1, c2 = st.columns([1, 3])
                with c1:
                    if not reviewed:
                        if st.button("Mark as reviewed", key=f"mark_{idx}"):
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
                        if st.button("Unmark reviewed", key=f"unmark_{idx}"):
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
                        with st.form(f"manual_edit_form_{idx}"):
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
                                    key = f"edit_c_{idx}"
                                )
                                selected_risks = [choice[1]]
                            col1, col2, col3, col4, col5, col6, col7 =  st.columns(7)
                            with col1:
                                upd_recency_value = st.number_input('Recency Risk', min_value = 0.0, max_value = 5.0, step = 1.0, value= float(article['Recency_Upd'] if pd.notna(article['Recency_Upd']) else article['Recency']), key =f"recency_input_{idx}")
                            with col2:
                                upd_acceleration_value = st.number_input('Acceleration Risk',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(article['Acceleration_value_Upd'] if pd.notna(article['Acceleration_value_Upd']) else article['Acceleration_value']),key =f"acceleration_input_{idx}")
                            with col3:
                                upd_source_accuracy =st.number_input('Source Accuracy',  min_value=0.0, max_value = 5.0, step = 1.0, value= float(article['Source_Accuracy_Upd'] if pd.notna(article['Source_Accuracy_Upd']) else article['Source_Accuracy']),key =f"source_input_{idx}")
                            with col4:
                                upd_impact_score = st.number_input('Impact Score',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(article['Impact_Score_Upd'] if pd.notna(article['Impact_Score_Upd']) else article['Impact_Score']),key =f"impact_input_{idx}")
                            with col5:
                                upd_location=st.number_input('Location Risk',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(article['Location_Upd'] if pd.notna(article['Location_Upd']) else article['Location']),key =f"location_input_{idx}")
                            with col6:
                                upd_industry_risk = st.number_input('Industry Risk',  min_value=0.0, max_value = 5.0, step = 1.0, value=float(article['Industry_Risk_Upd'] if pd.notna(article['Industry_Risk_Upd']) else article['Industry_Risk']),key =f"industry_input_{idx}")
                            with col7:
                                upd_frequency_score = st.number_input('Frequency Score', min_value=0.0, max_value = 5.0, step = 1.0, value=float(article['Frequency_Score_Upd'] if pd.notna(article['Frequency_Score_Upd']) else article['Frequency_Score']),key =f"frequency_input_{idx}")

                            st.markdown('Please provide a reason for the changes made to the risk labels:')
                            reason = st.text_area("Reason for changes", placeholder="Explain the changes made to the risk labels.", key=f"reason_{idx}")
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
