"""
Beacon — Streamlit + LangChain (modern) AI News Search (app.py) — final stable release

Highlights:
- No mutation of widget-backed session_state keys after widget creation (except inside callbacks).
- Trending chips use on_click callback that sets topic and optionally fetches immediately.
- Search button triggers fetch.
- Prev/Next use on_click callbacks that update page (no experimental_rerun).
- Pagination and footer centered per request.
"""

import os
import time
import math
import requests
import streamlit as st
from typing import List, Dict
from collections import Counter, defaultdict
import re
from html import unescape
import heapq

# optional dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# optional libs
try:
    import feedparser
except Exception:
    feedparser = None

# Gemini (google generative ai)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# native SDKs
try:
    import openai
except Exception:
    openai = None

try:
    import anthropic
except Exception:
    anthropic = None

# LangChain placeholders (best-effort imports)
LLMChain = None
PromptTemplate = None
LCOpenAI = None
LCAnthropic = None
LCVertexAI = None
LangChainOpenAI = None
try:
    from langchain import OpenAI as LangChainOpenAI
except Exception:
    LangChainOpenAI = None
try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
except Exception:
    try:
        from langchain import LLMChain, PromptTemplate
    except Exception:
        LLMChain = None
        PromptTemplate = None
try:
    from langchain_openai import OpenAI as LCOpenAI
except Exception:
    LCOpenAI = None
try:
    from langchain_anthropic import ChatAnthropic as LCAnthropic
except Exception:
    try:
        from langchain_anthropic import Anthropic as LCAnthropic
    except Exception:
        LCAnthropic = None
try:
    from langchain_google_vertexai import VertexAI as LCVertexAI
except Exception:
    LCVertexAI = None

# ---------------------------
# Helpers
# ---------------------------
def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    text = re.sub(r'<[^>]*>', '', raw_html)
    text = unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def local_extractive_summary(articles: List[Dict], max_summary_sentences: int = 3, takeaways: int = 3) -> str:
    texts = []
    for a in articles:
        t = clean_html(a.get('title') or '')
        d = clean_html(a.get('description') or '')
        if d:
            texts.append(t + '. ' + d)
        elif t:
            texts.append(t)
    if not texts:
        return "Summary: (no content)\n\nTakeaways:\n- (no content)"

    raw = " ".join(texts)
    sentences = re.split(r'(?<=[.!?])\s+', raw)
    stopwords = set([
        'the','a','an','of','and','to','in','on','for','with','at','from','by','about',
        'after','before','against','is','are','was','were','it','this','that','as','be',
        'has','have','had','will','would','can','could','should','i','you','they','we'
    ])
    freq = defaultdict(int)
    for s in sentences:
        for w in re.findall(r"\w+", s.lower()):
            if w in stopwords or len(w) <= 2:
                continue
            freq[w] += 1

    if not freq:
        summary_text = " ".join(sentences[:max_summary_sentences]).strip()
        return f"Summary: {summary_text}\n\nTakeaways:\n- (no keywords found)"

    sent_scores = []
    for s in sentences:
        score = sum(freq.get(w, 0) for w in re.findall(r"\w+", s.lower()))
        sent_scores.append((score, s))

    top_sents = heapq.nlargest(max_summary_sentences, sent_scores, key=lambda x: x[0])
    selected = [s for (_, s) in top_sents]
    selected_sorted = sorted(selected, key=lambda s: raw.find(s))
    summary_text = " ".join([s.strip() for s in selected_sorted]).strip()

    top_keywords = heapq.nlargest(takeaways, freq.items(), key=lambda x: x[1])
    takeaways_list = [kw.replace('_', ' ').capitalize() for kw, _ in top_keywords]

    formatted = f"Summary: {summary_text}\n\nTakeaways:\n"
    for k in takeaways_list:
        formatted += f"- {k}\n"
    return formatted

# ---------------------------
# Network helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def get_trending_topics(limit: int = 8, region: str = "india", rss_only: bool = False) -> List[str]:
    if not rss_only:
        try:
            from pytrends.request import TrendReq
            pytrends = TrendReq(hl='en-US', tz=330)
            df = pytrends.trending_searches(pn=region)
            if df is not None and not df.empty:
                return list(df[0].head(limit))
        except Exception:
            pass
        try:
            key = os.getenv('NEWSAPI_KEY')
            if key:
                url = 'https://newsapi.org/v2/top-headlines'
                params = {'apiKey': key, 'language': 'en', 'pageSize': 50}
                r = requests.get(url, params=params, timeout=8)
                r.raise_for_status()
                data = r.json()
                titles = [a.get('title','') for a in data.get('articles', [])]
                words = Counter()
                stopwords = set([w.strip().lower() for w in "the a an of and to in on for with at from by about after before against".split()])
                for t in titles:
                    for w in t.split():
                        w2 = ''.join(ch for ch in w if ch.isalpha()).lower()
                        if len(w2) > 3 and w2 not in stopwords:
                            words[w2] += 1
                most = [w.capitalize() for w,_ in words.most_common(limit)]
                if most:
                    return most
        except Exception:
            pass
    if feedparser is not None:
        try:
            feed = feedparser.parse('https://news.google.com/rss')
            titles = [e.get('title','') for e in feed.entries[:120]]
            words = Counter()
            for t in titles:
                for w in t.split():
                    w2 = ''.join(ch for ch in w if ch.isalpha()).lower()
                    if len(w2) > 3:
                        words[w2] += 1
            most = [w.capitalize() for w,_ in words.most_common(limit)]
            if most:
                return most
        except Exception:
            pass
    return ["AI","Climate","Elections","Startups","Tech","Crypto","Movies","Sports"][:limit]

@st.cache_data(show_spinner=False)
def fetch_news_with_newsapi(topic: str, page_size: int = 10) -> List[Dict]:
    key = os.getenv('NEWSAPI_KEY')
    if not key:
        return []
    try:
        url = 'https://newsapi.org/v2/everything'
        params = {'q': topic, 'language': 'en', 'sortBy': 'relevancy', 'pageSize': page_size, 'apiKey': key}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get('articles', [])
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def fetch_news_with_rss(topic: str, page_size: int = 10) -> List[Dict]:
    if feedparser is None:
        return []
    try:
        q = requests.utils.requote_uri(topic)
        url = f'https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en'
        feed = feedparser.parse(url)
        results = []
        for e in feed.entries[:page_size]:
            results.append({
                'title': e.get('title'),
                'description': e.get('summary', ''),
                'url': e.get('link'),
                'source': {'name': e.get('source', {}).get('title') if isinstance(e.get('source'), dict) else ''},
                'publishedAt': e.get('published') or ''
            })
        return results
    except Exception:
        return []

def fetch_news_for_topic(topic: str, page_size: int = 10, rss_only: bool = False) -> List[Dict]:
    if not rss_only:
        articles = fetch_news_with_newsapi(topic, page_size)
        if articles:
            return articles
    return fetch_news_with_rss(topic, page_size)

# ---------------------------
# LLM loader / summarizer (kept minimal)
# ---------------------------
def get_llm_wrapper(model_choice: str):
    # Minimal loader — preserved from earlier; returns wrapper or None
    mc = (model_choice or "").lower()
    if mc == 'gemini':
        if genai is None:
            st.warning('Gemini client not installed.')
            return None
        key = os.getenv('GEMINI_API_KEY')
        if not key:
            st.warning('GEMINI_API_KEY not set.')
            return None
        try:
            genai.configure(api_key=key)
            return {'provider': 'gemini', 'model': os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5')}
        except Exception as e:
            st.warning(f'Gemini init failed: {e}')
            return None
    if mc == 'chatgpt':
        try:
            from langchain import OpenAI as NewLCOpenAI
            key = os.getenv('OPENAI_API_KEY')
            if key:
                try:
                    return NewLCOpenAI(openai_api_key=key)
                except TypeError:
                    return NewLCOpenAI(api_key=key)
        except Exception:
            pass
        if LCOpenAI is not None:
            key = os.getenv('OPENAI_API_KEY')
            if key:
                try:
                    return LCOpenAI(api_key=key)
                except TypeError:
                    return LCOpenAI(openai_api_key=key)
        if openai is not None:
            key = os.getenv('OPENAI_API_KEY')
            if key:
                openai.api_key = key
                return {'provider': 'openai', 'client': openai}
        st.warning('No ChatGPT support available.')
        return None
    if mc == 'claude':
        if LCAnthropic is not None:
            key = os.getenv('ANTHROPIC_API_KEY')
            if key:
                try:
                    return LCAnthropic(api_key=key)
                except Exception as e:
                    st.warning(f'LangChain Anthropic init failed: {e}')
        if anthropic is not None:
            key = os.getenv('ANTHROPIC_API_KEY')
            if key:
                try:
                    client = anthropic.Client(api_key=key)
                    return {'provider': 'anthropic', 'client': client}
                except Exception as e:
                    st.warning(f'Anthropic init failed: {e}')
        st.warning('No Claude support available.')
        return None
    return None

def summarize_with_llm_or_local(llm, articles: List[Dict]) -> str:
    if not articles:
        return local_extractive_summary([], 3, 3)
    if llm is None:
        return local_extractive_summary(articles, 3, 3)
    # Minimal LLM usage preserved — use local fallback if anything fails
    try:
        if isinstance(llm, dict) and llm.get('provider') == 'openai' and openai is not None:
            client = llm['client']
            parts = [f"{i+1}. {clean_html(a.get('title',''))} — {clean_html(a.get('description',''))}" for i,a in enumerate(articles[:8])]
            prompt = "Summarize these news articles in 3 sentences and give 3 key takeaways:\n\n" + "\n".join(parts)
            if hasattr(client, "ChatCompletion"):
                resp = client.ChatCompletion.create(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                                                   messages=[{"role":"user","content":prompt}],
                                                   max_tokens=400, temperature=0.2)
                if isinstance(resp, dict):
                    choices = resp.get("choices")
                    if choices:
                        return choices[0].get("message", {}).get("content") or choices[0].get("text","")
                return str(resp)
            else:
                resp = client.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=400, temperature=0.2)
                return resp.choices[0].text
    except Exception:
        pass
    return local_extractive_summary(articles, max_summary_sentences=3, takeaways=3)

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title='Automated Knowledge Collector', layout='wide')

# center title
t1, t2, t3 = st.columns([1, 2, 1])
with t2:
    st.markdown("<h1 style='text-align:center'>Automated News Collector</h1>", unsafe_allow_html=True)
    st.caption("Search topics and get a combined summary. Use RSS-only mode if you don't have API keys.")

# ensure session_state keys exist BEFORE widget creation
if 'topic_input' not in st.session_state:
    st.session_state['topic_input'] = ''
if 'page' not in st.session_state:
    st.session_state['page'] = 1
if 'articles' not in st.session_state:
    st.session_state['articles'] = []
if 'articles_topic' not in st.session_state:
    st.session_state['articles_topic'] = ''
# pages will be set after computing total/pages

# central search area
with st.container():
    cl, cm, cr = st.columns([1, 3, 1])
    with cm:
        topic_input = st.text_input('Enter topic', value=st.session_state.get('topic_input',''),
                                    placeholder='e.g. artificial intelligence, climate change',
                                    key='topic_input')
        col1, col2, col3 = st.columns([2,2,2])
        with col1:
            model_choice = st.selectbox('Model', ['ChatGPT','Claude','Gemini'])
        with col2:
            page_size = st.selectbox('Articles/page', [5,7,10,15], index=1)
        with col3:
            rss_only = st.checkbox('RSS-only', value=False, help='Use RSS if you have no API keys')
        run_search = st.button('Search', key='search_center')

# callback to set topic and immediately fetch articles (safe — runs before rerun)
def _set_topic_and_fetch(topic_value):
    st.session_state['topic_input'] = topic_value
    st.session_state['page'] = 1
    # immediate fetch: keep it reasonably sized to avoid blocking too long
    try:
        articles = fetch_news_for_topic(topic_value, page_size=50 if page_size > 15 else page_size*3, rss_only=rss_only)
    except Exception:
        articles = []
    st.session_state['articles'] = articles or []
    st.session_state['articles_topic'] = topic_value
    # no experimental_rerun() — Streamlit will rerun automatically after callback

# trending chips
trending = get_trending_topics(limit=8, rss_only=rss_only)
st.markdown("<div style='text-align:center'><b>Trending:</b></div>", unsafe_allow_html=True)
tcols = st.columns(len(trending))
for i, t in enumerate(trending):
    tcols[i].button(t, key=f"trend_{i}", on_click=_set_topic_and_fetch, args=(t,))

# Decide whether to fetch (Search button clicked or topic changed)
effective_topic = (st.session_state.get('topic_input') or '').strip()
need_fetch = False
if run_search:
    need_fetch = True
else:
    if effective_topic and st.session_state.get('articles_topic') != effective_topic:
        # no immediate fetch requested, but we want to auto-fetch once when topic changes via other means
        need_fetch = True

if need_fetch and effective_topic:
    with st.spinner('Fetching articles...'):
        articles = fetch_news_for_topic(effective_topic, page_size=50 if page_size > 15 else page_size*3, rss_only=rss_only)
        time.sleep(0.12)
    st.session_state['articles'] = articles or []
    st.session_state['articles_topic'] = effective_topic

articles = st.session_state.get('articles', [])

if not effective_topic and not articles:
    st.info('Enter a topic or click a trending topic.')

if articles:
    per_page = page_size
    total = len(articles)
    pages = max(1, math.ceil(total / per_page))
    st.session_state['pages'] = pages

    # clamp page
    if st.session_state.page < 1:
        st.session_state.page = 1
    if st.session_state.page > pages:
        st.session_state.page = pages

    page = st.session_state.page
    start = (page - 1) * per_page
    end = min(total, start + per_page)
    page_articles = articles[start:end]

    # pagination callbacks (no experimental_rerun)
    def _go_prev():
        if st.session_state.get('page', 1) > 1:
            st.session_state.page -= 1
    def _go_next():
        if st.session_state.get('page', 1) < st.session_state.get('pages', 1):
            st.session_state.page += 1

    # centered top pagination
    left, center, right = st.columns([1, 2, 1])
    with center:
        p1, p2, p3 = st.columns([1,2,1])
        with p1:
            st.button('Prev', key='prev_top', on_click=_go_prev)
        with p2:
            st.markdown(f"<div style='text-align:center; font-weight:600'>Page {st.session_state.page} / {pages} — {total} results</div>", unsafe_allow_html=True)
        with p3:
            st.button('Next', key='next_top', on_click=_go_next)

    # prepare LLM wrapper (best-effort)
    llm = None
    if model_choice:
        llm = get_llm_wrapper(model_choice)

    # combined summary
    with st.expander('Combined summary & takeaways'):
        with st.spinner('Summarizing...'):
            combined_text = summarize_with_llm_or_local(llm, page_articles)
            st.info(clean_html(combined_text))

    # show articles
    for idx, a in enumerate(page_articles, start=start+1):
        st.markdown(f"### {idx}. {clean_html(a.get('title',''))}")
        src = a.get('source') or {}
        src_name = src.get('name') if isinstance(src, dict) else src or ''
        pub = (a.get('publishedAt') or '')[:19]
        st.write(f"**{src_name}** — {pub}")
        desc = clean_html(a.get('description') or '')
        if desc:
            st.write(desc)
        url = a.get('url') or ''
        if url:
            st.markdown(f"[Read full article]({url})")

    # centered bottom pagination (same callbacks)
    bl, bc, br = st.columns([1, 2, 1])
    with bc:
        bp1, bp2, bp3 = st.columns([1,2,1])
        with bp1:
            st.button('Prev', key='prev_bottom', on_click=_go_prev)
        with bp2:
            st.markdown(f"<div style='text-align:center; font-weight:600'>Page {st.session_state.page} / {pages}</div>", unsafe_allow_html=True)
        with bp3:
            st.button('Next', key='next_bottom', on_click=_go_next)

# centered footer
st.markdown('---')
st.markdown("<div style='text-align:center'><small>Thanks for using our services.</small></div>", unsafe_allow_html=True)
