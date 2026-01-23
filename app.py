"""
Analyst Alpha Dashboard - Fast Version (Reads from pre-calculated data)
========================================================================

Runs significantly faster by reading from pre-calculated alpha_index.db
instead of calculating on the fly.

Usage:
1. First run: python precalculate.py
2. Then run: streamlit run app_fast.py
"""
import streamlit as st
import pandas as pd
import sqlite3
import os
import sys
import plotly.graph_objects as go
from datetime import datetime
import gdown

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    COLORS, 
    DEFAULT_START_DATE, 
    DEFAULT_END_DATE,
    VNINDEX_TICKER,
    SKIP_RATINGS
)
from components.charts import (
    create_alpha_vs_vnindex_chart, create_daily_alpha_bars,
    create_ranking_bars, create_team_overview_chart,
    create_peer_alpha_chart
)
from data.database import get_vnindex_prices
from data.calculations import (
    calculate_analyst_summary,
    calculate_analyst_alpha_index,
    calculate_analyst_daily_alpha
)

# Pre-calculated database path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALPHA_INDEX_DB = os.path.join(BASE_DIR, 'alpha_index.db')
DRIVE_FILE_ID = "1KQYKwD_DPYbECN8iUmiLnXuEYD_vqnFh"

def sync_db_from_drive():
    """Download fresh alpha_index.db from Google Drive on startup."""
    # Use st.cache_resource to avoid redownloading on every rerun/interaction
    # But we want it to run once per session start.
    if 'db_synced' not in st.session_state:
        st.toast("Syncing data from Cloud...", icon="☁️")
        try:
            url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
            # Always overwrite
            gdown.download(url, ALPHA_INDEX_DB, quiet=False)
            st.session_state['db_synced'] = True
            st.toast("Data synced successfully!", icon="✅")
        except Exception as e:
            st.error(f"Failed to sync database: {e}")
            # Do not stop if file exists locally (fallback)
            if not os.path.exists(ALPHA_INDEX_DB):
                st.stop()

# Sync DB on startup
# sync_db_from_drive()


# Page configuration
st.set_page_config(
    page_title="Analyst Alpha Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .fast-badge {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)


def check_database_exists():
    """Check if pre-calculated database exists."""
    return os.path.exists(ALPHA_INDEX_DB)


@st.cache_data(ttl=3000)
def load_scorecard() -> pd.DataFrame:
    """Load analyst scorecard from pre-calculated database."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    # Get the latest as_of_date
    latest_date = pd.read_sql_query(
        "SELECT MAX(as_of_date) as max_date FROM AnalystSummary", conn
    ).iloc[0]['max_date']
    
    # Load scorecard for latest date
    df = pd.read_sql_query("""
        SELECT 
            analyst_email,
            analyst_name,
            index_value,
            ytd_alpha,
            hit_rate,
            information_ratio,
            conviction,
            coverage,
            opf_count,
            upf_count,
            mpf_count
        FROM AnalystSummary
        WHERE as_of_date = ?
        ORDER BY index_value DESC
    """, conn, params=(latest_date,))
    
    conn.close()
    
    # Add rank
    df['rank'] = range(1, len(df) + 1)
    
    return df


@st.cache_data(ttl=3000)
def load_analyst_history(analyst_email: str) -> pd.DataFrame:
    """Load alpha history for a specific analyst with active tickers."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    df = pd.read_sql_query("""
        WITH TickerList AS (
            SELECT trade_date, 
                   GROUP_CONCAT(ticker || ' ' || recommendation, '<br>') as active_tickers
            FROM DailyContributions
            WHERE analyst_email = ?
            GROUP BY trade_date
        )
        SELECT 
            t1.trade_date as date,
            t1.daily_alpha,
            t1.index_value,
            t1.hits,
            t1.total,
            t1.cumulative_hits,
            t1.cumulative_total,
            t2.active_tickers
        FROM AnalystAlphaDaily t1
        LEFT JOIN TickerList t2 ON t1.trade_date = t2.trade_date
        WHERE t1.analyst_email = ?
        ORDER BY t1.trade_date
    """, conn, params=(analyst_email, analyst_email))
    
    conn.close()
    return df


@st.cache_data(ttl=3000)
def load_analyst_ticker_list(analyst_email: str) -> pd.DataFrame:
    """Load list of active tickers and their contributions per date."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    # Fetch raw data with priority ordering
    df = pd.read_sql_query("""
        SELECT 
            trade_date as date, 
            ticker,
            recommendation,
            CASE 
                WHEN recommendation IN ('OUTPERFORM', 'BUY') THEN 1
                WHEN recommendation = 'MARKET-PERFORM' THEN 2
                WHEN recommendation IN ('UNDER-PERFORM', 'UNDERPERFORM', 'AVOID', 'SELL') THEN 3
                ELSE 4
            END as rec_priority
        FROM DailyContributions
        WHERE analyst_email = ?
        ORDER BY trade_date, rec_priority, ticker
    """, conn, params=(analyst_email,))
    
    conn.close()
    
    if df.empty:
        return pd.DataFrame(columns=['date', 'active_tickers'])
        
    # Format: "Ticker   Recommendation" with fixed-width ticker for alignment
    df['info'] = df.apply(lambda x: f"{x['ticker'].ljust(5).replace(' ', '&nbsp;')}{x['recommendation']}", axis=1)
    
    # Group by date and join with HTML break
    grouped = df.groupby('date')['info'].apply(lambda x: '<br>'.join(x)).reset_index()
    grouped.columns = ['date', 'active_tickers']
    
    return grouped


@st.cache_data(ttl=3000)
def load_meta_info() -> dict:
    """Load calculation metadata."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    df = pd.read_sql_query("""
        SELECT * FROM CalculationMeta ORDER BY id DESC LIMIT 1
    """, conn)
    
    conn.close()
    
    if df.empty:
        return None
    
    return df.iloc[0].to_dict()


@st.cache_data(ttl=3000)
def load_vnindex_normalized(start_date: str, end_date: str) -> pd.DataFrame:
    """Load VnIndex normalized to 100."""
    # from data.database import get_stock_prices
    # prices = get_stock_prices(VNINDEX_TICKER, start_date, end_date)
    
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    prices = pd.read_sql_query("""
        SELECT trade_date as TradeDate, adj_close as AdjClose
        FROM VnIndexPrices
        WHERE trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """, conn, params=(start_date, end_date))
    conn.close()
    
    if prices.empty:
        return pd.DataFrame()
    
    # Normalize to 1
    first_price = prices['AdjClose'].iloc[0]
    prices['index_value'] = prices['AdjClose'] / first_price
    prices = prices.rename(columns={'TradeDate': 'date'})
    
    return prices[['date', 'index_value']]


@st.cache_data(ttl=3000)
def load_daily_contributions(analyst_email: str, trade_date: str = None) -> pd.DataFrame:
    """Load stock-level contributions for an analyst on a specific date."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    if trade_date:
        df = pd.read_sql_query("""
            SELECT 
                trade_date,
                ticker,
                recommendation,
                direction_weight,
                stock_return,
                vnindex_return,
                excess_return,
                contribution,
                is_correct
            FROM DailyContributions
            WHERE analyst_email = ? AND trade_date = ?
            ORDER BY contribution DESC
        """, conn, params=(analyst_email, trade_date))
    else:
        # Get all dates
        df = pd.read_sql_query("""
            SELECT 
                trade_date,
                ticker,
                recommendation,
                direction_weight,
                stock_return,
                vnindex_return,
                excess_return,
                contribution,
                is_correct
            FROM DailyContributions
            WHERE analyst_email = ?
            ORDER BY trade_date DESC, contribution DESC
        """, conn, params=(analyst_email,))
    
    conn.close()
    return df


@st.cache_data(ttl=3000)
def get_available_dates(analyst_email: str) -> list:
    """Get list of available trading dates for an analyst."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    df = pd.read_sql_query("""
        SELECT DISTINCT trade_date 
        FROM DailyContributions 
        WHERE analyst_email = ?
        ORDER BY trade_date DESC
    """, conn, params=(analyst_email,))
    conn.close()
    return df['trade_date'].tolist()


@st.cache_data(ttl=3000)
def get_available_dates_peer(analyst_email: str) -> list:
    """Get list of available trading dates for an analyst (peer contributions)."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    df = pd.read_sql_query("""
        SELECT DISTINCT trade_date 
        FROM PeerDailyContributions 
        WHERE analyst_email = ?
        ORDER BY trade_date DESC
    """, conn, params=(analyst_email,))
    conn.close()
    return df['trade_date'].tolist()


# ===== PEER COMPARISON DATA LOADING FUNCTIONS =====

@st.cache_data(ttl=3000)
def load_peer_scorecard() -> pd.DataFrame:
    """Load analyst scorecard from peer-based calculation."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    # Get the latest as_of_date
    latest_date = pd.read_sql_query(
        "SELECT MAX(as_of_date) as max_date FROM PeerAnalystSummary", conn
    ).iloc[0]['max_date']
    
    # Load scorecard for latest date
    df = pd.read_sql_query("""
        SELECT 
            analyst_email,
            analyst_name,
            index_value,
            ytd_alpha,
            hit_rate,
            information_ratio,
            conviction,
            coverage,
            opf_count,
            upf_count,
            mpf_count
        FROM PeerAnalystSummary
        WHERE as_of_date = ?
        ORDER BY index_value DESC
    """, conn, params=(latest_date,))
    
    conn.close()
    
    # Add rank
    df['rank'] = range(1, len(df) + 1)
    
    return df


@st.cache_data(ttl=3000)
def load_peer_analyst_history(analyst_email: str) -> pd.DataFrame:
    """Load peer-based alpha history for a specific analyst with active tickers."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    df = pd.read_sql_query("""
        WITH TickerList AS (
            SELECT trade_date, 
                   GROUP_CONCAT(ticker || ' ' || recommendation, '<br>') as active_tickers
            FROM PeerDailyContributions
            WHERE analyst_email = ?
            GROUP BY trade_date
        )
        SELECT 
            t1.trade_date as date,
            t1.daily_alpha,
            t1.index_value,
            t1.hits,
            t1.total,
            t1.cumulative_hits,
            t1.cumulative_total,
            t2.active_tickers
        FROM PeerAnalystAlphaDaily t1
        LEFT JOIN TickerList t2 ON t1.trade_date = t2.trade_date
        WHERE t1.analyst_email = ?
        ORDER BY t1.trade_date
    """, conn, params=(analyst_email, analyst_email))
    
    conn.close()
    return df


@st.cache_data(ttl=3000)
def load_peer_daily_contributions(analyst_email: str, trade_date: str = None) -> pd.DataFrame:
    """Load peer-based stock-level contributions for an analyst."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    if trade_date:
        df = pd.read_sql_query("""
            SELECT 
                trade_date,
                ticker,
                recommendation,
                direction_weight,
                stock_return,
                peer_return,
                excess_return,
                contribution,
                is_correct
            FROM PeerDailyContributions
            WHERE analyst_email = ? AND trade_date = ?
            ORDER BY contribution DESC
        """, conn, params=(analyst_email, trade_date))
    else:
        df = pd.read_sql_query("""
            SELECT 
                trade_date,
                ticker,
                recommendation,
                direction_weight,
                stock_return,
                peer_return,
                excess_return,
                contribution,
                is_correct
            FROM PeerDailyContributions
            WHERE analyst_email = ?
            ORDER BY trade_date DESC, contribution DESC
        """, conn, params=(analyst_email,))
    
    conn.close()
    return df


@st.cache_data(ttl=3000)
def get_peer_available_dates(analyst_email: str) -> list:
    """Get list of available trading dates for peer comparison."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    df = pd.read_sql_query("""
        SELECT DISTINCT trade_date 
        FROM PeerDailyContributions 
        WHERE analyst_email = ?
        ORDER BY trade_date DESC
    """, conn, params=(analyst_email,))
    conn.close()
    return df['trade_date'].tolist()


# ===== INDUSTRY ALPHA DATA LOADING FUNCTIONS =====

@st.cache_data(ttl=3000)
def load_industry_scorecard() -> pd.DataFrame:
    """Load industry scorecard from pre-calculated database (vs VnIndex)."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    table_check = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='IndustrySummary'", conn
    )
    if table_check.empty:
        conn.close()
        return pd.DataFrame()
    
    latest_date = pd.read_sql_query(
        "SELECT MAX(as_of_date) as max_date FROM IndustrySummary", conn
    ).iloc[0]['max_date']
    
    df = pd.read_sql_query("""
        SELECT industry, index_value, ytd_alpha, hit_rate, information_ratio,
               coverage, opf_count, upf_count, mpf_count
        FROM IndustrySummary WHERE as_of_date = ? ORDER BY index_value DESC
    """, conn, params=(latest_date,))
    conn.close()
    
    if not df.empty:
        df['rank'] = range(1, len(df) + 1)
    return df


@st.cache_data(ttl=3000)
def load_industry_history(industry: str) -> pd.DataFrame:
    """Load alpha history for a specific industry (vs VnIndex) with active tickers."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    df = pd.read_sql_query("""
        WITH TickerList AS (
            SELECT trade_date, 
                   GROUP_CONCAT(ticker || ' ' || recommendation, '<br>') as active_tickers
            FROM IndustryDailyContributions
            WHERE industry = ?
            GROUP BY trade_date
        )
        SELECT t1.trade_date as date, t1.daily_alpha, t1.index_value, t1.hits, t1.total,
               t1.cumulative_hits, t1.cumulative_total,
               t2.active_tickers
        FROM IndustryAlphaDaily t1
        LEFT JOIN TickerList t2 ON t1.trade_date = t2.trade_date
        WHERE t1.industry = ? 
        ORDER BY t1.trade_date
    """, conn, params=(industry, industry))
    conn.close()
    return df


@st.cache_data(ttl=3000)
def load_industry_daily_contributions(industry: str, trade_date: str = None) -> pd.DataFrame:
    """Load stock-level contributions for an industry."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    if trade_date:
        df = pd.read_sql_query("""
            SELECT trade_date, ticker, recommendation, direction_weight, stock_return,
                   vnindex_return, excess_return, contribution, is_correct
            FROM IndustryDailyContributions WHERE industry = ? AND trade_date = ?
            ORDER BY contribution DESC
        """, conn, params=(industry, trade_date))
    else:
        df = pd.read_sql_query("""
            SELECT trade_date, ticker, recommendation, direction_weight, stock_return,
                   vnindex_return, excess_return, contribution, is_correct
            FROM IndustryDailyContributions WHERE industry = ?
            ORDER BY trade_date DESC, contribution DESC
        """, conn, params=(industry,))
    conn.close()
    return df


@st.cache_data(ttl=3000)
def load_industry_peer_scorecard() -> pd.DataFrame:
    """Load industry scorecard (vs Sector Average)."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    table_check = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='IndustryPeerSummary'", conn
    )
    if table_check.empty:
        conn.close()
        return pd.DataFrame()
    
    latest_date = pd.read_sql_query(
        "SELECT MAX(as_of_date) as max_date FROM IndustryPeerSummary", conn
    ).iloc[0]['max_date']
    
    df = pd.read_sql_query("""
        SELECT industry, index_value, ytd_alpha, hit_rate, information_ratio,
               coverage, opf_count, upf_count, mpf_count
        FROM IndustryPeerSummary WHERE as_of_date = ? ORDER BY index_value DESC
    """, conn, params=(latest_date,))
    conn.close()
    
    if not df.empty:
        df['rank'] = range(1, len(df) + 1)
    return df


@st.cache_data(ttl=3000)
def load_industry_peer_history(industry: str) -> pd.DataFrame:
    """Load alpha history for an industry (vs Sector Average) with active tickers."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    df = pd.read_sql_query("""
        WITH TickerList AS (
            SELECT trade_date, 
                   GROUP_CONCAT(ticker || ' ' || recommendation, '<br>') as active_tickers
            FROM IndustryPeerDailyContributions
            WHERE industry = ?
            GROUP BY trade_date
        )
        SELECT t1.trade_date as date, t1.daily_alpha, t1.index_value, t1.hits, t1.total,
               t1.cumulative_hits, t1.cumulative_total,
               t2.active_tickers
        FROM IndustryPeerAlphaDaily t1
        LEFT JOIN TickerList t2 ON t1.trade_date = t2.trade_date
        WHERE t1.industry = ? 
        ORDER BY t1.trade_date
    """, conn, params=(industry, industry))
    conn.close()
    return df


@st.cache_data(ttl=3000)
def load_industry_peer_contributions(industry: str, trade_date: str = None) -> pd.DataFrame:
    """Load stock-level contributions for an industry (vs Sector Average)."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    if trade_date:
        df = pd.read_sql_query("""
            SELECT trade_date, ticker, recommendation, direction_weight, stock_return,
                   sector_return, excess_return, contribution, is_correct
            FROM IndustryPeerDailyContributions WHERE industry = ? AND trade_date = ?
            ORDER BY contribution DESC
        """, conn, params=(industry, trade_date))
    else:
        df = pd.read_sql_query("""
            SELECT trade_date, ticker, recommendation, direction_weight, stock_return,
                   sector_return, excess_return, contribution, is_correct
            FROM IndustryPeerDailyContributions WHERE industry = ?
            ORDER BY trade_date DESC, contribution DESC
        """, conn, params=(industry,))
    conn.close()
    return df


@st.cache_data(ttl=3000)
def get_industry_available_dates(industry: str, use_peer: bool = False) -> list:
    """Get list of available trading dates for an industry."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    table = "IndustryPeerDailyContributions" if use_peer else "IndustryDailyContributions"
    df = pd.read_sql_query(f"""
        SELECT DISTINCT trade_date FROM {table} WHERE industry = ?
        ORDER BY trade_date DESC
    """, conn, params=(industry,))
    conn.close()
    return df['trade_date'].tolist() if not df.empty else []


def main():
    """Main application."""
    st.markdown('<h1 class="main-header">📊 Analyst Alpha Dashboard <span class="fast-badge">⚡ Fast</span></h1>', 
                unsafe_allow_html=True)
    
    # Check if database exists
    if not check_database_exists():
        st.error("""
        ⚠️ **Pre-calculated data not found!**
        
        Please run the pre-calculation script first:
        ```bash
        python precalculate.py
        ```
        """)
        return
    
    # Load meta info
    meta = load_meta_info()
    if meta:
        st.sidebar.info(f"""
        📅 **Data Range:** {meta['start_date']} → {meta['end_date']}  
        🕐 **Last Updated:** {meta['last_updated']}  
        👥 **Analysts:** {meta['total_analysts']}
        """)
    
    # Load scorecard (cached, very fast)
    with st.spinner("Loading data..."):
        scorecard = load_scorecard()
        peer_scorecard = load_peer_scorecard()
    
    if scorecard.empty:
        st.warning("No analyst data available.")
        return
    
    # Navigation
    st.sidebar.title("⚙️ Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["📊 Analyst Alpha", "🏭 Industry Alpha", "📋 Calculation Detail", "🏆 Leaderboard"]
    )
    
    if page == "📊 Analyst Alpha":
        render_analyst_alpha(scorecard, peer_scorecard, meta)
    elif page == "🏭 Industry Alpha":
        render_industry_alpha(meta)
    elif page == "📋 Calculation Detail":
        render_calculation_detail(scorecard, peer_scorecard)
    elif page == "🏆 Leaderboard":
        render_leaderboard(scorecard)


def render_analyst_alpha(scorecard: pd.DataFrame, peer_scorecard: pd.DataFrame, meta: dict):
    """Render merged Analyst Alpha page with tabs."""
    st.subheader("📊 Analyst Alpha")
    
    tab1, tab2 = st.tabs(["📊 vs VnIndex", "🔄 vs Coverage"])
    
    with tab1:
        st.markdown("### Performance vs VnIndex")
        render_analyst_alpha_tab(scorecard, meta, use_peer=False)
        
    with tab2:
        st.markdown("### Performance vs Sector Average")
        render_analyst_alpha_tab(peer_scorecard, meta, use_peer=True)


@st.cache_data(ttl=3000)
def load_daily_scorecard(trade_date: str) -> pd.DataFrame:
    """Load scorecard (rankings) for a specific date (vs VnIndex) with calculated metrics."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    # 1. Base Scorecard Data
    df = pd.read_sql_query("""
        SELECT analyst_email, index_value, cumulative_hits, cumulative_total
        FROM AnalystAlphaDaily 
        WHERE trade_date = ?
    """, conn, params=(trade_date,))
    
    if df.empty:
        conn.close()
        return pd.DataFrame()
        
    # Standardize email for merging
    df['email_key'] = df['analyst_email'].str.lower().str.strip()

    # 2. Coverage & Conviction (from DailyContributions on that date)
    contrib_df = pd.read_sql_query("""
        SELECT analyst_email, direction_weight
        FROM DailyContributions
        WHERE trade_date = ?
    """, conn, params=(trade_date,))
    
    metrics = []
    if not contrib_df.empty:
        for email, group in contrib_df.groupby('analyst_email'):
            total = len(group)
            opf = len(group[group['direction_weight'] > 0])
            upf = len(group[group['direction_weight'] == -1.0])
            
            conviction = ((opf + upf) / total * 100) if total > 0 else 0
            metrics.append({
                'email_key': email.lower().strip(),
                'coverage': total,
                'conviction': conviction
            })
    
    if metrics:
        metrics_df = pd.DataFrame(metrics)
    else:
        metrics_df = pd.DataFrame(columns=['email_key', 'coverage', 'conviction'])
    
    # 3. Information Ratio (History up to date)
    history_df = pd.read_sql_query("""
        SELECT analyst_email, daily_alpha
        FROM AnalystAlphaDaily
        WHERE trade_date <= ?
    """, conn, params=(trade_date,))
    
    ir_metrics = []
    if not history_df.empty:
        for email, group in history_df.groupby('analyst_email'):
            if len(group) >= 20:
                avg = group['daily_alpha'].mean()
                std = group['daily_alpha'].std()
                ir = avg / std if std > 0 else 0
            else:
                ir = None
            
            ir_metrics.append({
                'email_key': email.lower().strip(),
                'information_ratio': ir
            })
            
    if ir_metrics:
        ir_df = pd.DataFrame(ir_metrics)
    else:
        ir_df = pd.DataFrame(columns=['email_key', 'information_ratio'])

    conn.close()
    
    # Merge all on email_key
    df = df.merge(metrics_df, on='email_key', how='left')
    df = df.merge(ir_df, on='email_key', how='left')
    
    # Fill N/A for Coverage/Conviction if missing (implies 0 activity)
    df['coverage'] = df['coverage'].fillna(0)
    # If coverage is 0, conviction is N/A or 0. Let's set to N/A if coverage is 0? 
    # Or just keep NaN and let formatter handle it.
    
    # Cleanup keys
    if 'email_key' in df.columns:
        df = df.drop(columns=['email_key'])
        
    # Calculate derived metrics
    df['total_alpha'] = (df['index_value'] - 1) * 100
    df['hit_rate'] = (df['cumulative_hits'] / df['cumulative_total'] * 100).fillna(0)
    
    # Sort
    df = df.sort_values('index_value', ascending=False)
    df['rank'] = range(1, len(df) + 1)
    
    return df

@st.cache_data(ttl=3000)
def load_peer_daily_scorecard(trade_date: str) -> pd.DataFrame:
    """Load scorecard (rankings) for a specific date (vs Peer) with metrics."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    # 1. Base Scorecard
    df = pd.read_sql_query("""
        SELECT analyst_email, index_value, cumulative_hits, cumulative_total
        FROM PeerAnalystAlphaDaily 
        WHERE trade_date = ?
    """, conn, params=(trade_date,))
    
    if df.empty:
        conn.close()
        return pd.DataFrame()

    # Standardize email
    df['email_key'] = df['analyst_email'].str.lower().str.strip()

    # 2. Coverage & Conviction
    contrib_df = pd.read_sql_query("""
        SELECT analyst_email, direction_weight
        FROM PeerDailyContributions
        WHERE trade_date = ?
    """, conn, params=(trade_date,))
    
    metrics = []
    if not contrib_df.empty:
        for email, group in contrib_df.groupby('analyst_email'):
            total = len(group)
            opf = len(group[group['direction_weight'] > 0])
            upf = len(group[group['direction_weight'] == -1.0])
            
            conviction = ((opf + upf) / total * 100) if total > 0 else 0
            metrics.append({
                'email_key': email.lower().strip(),
                'coverage': total,
                'conviction': conviction
            })
    
    if metrics:
        metrics_df = pd.DataFrame(metrics)
    else:
        metrics_df = pd.DataFrame(columns=['email_key', 'coverage', 'conviction'])
    
    # 3. Information Ratio
    history_df = pd.read_sql_query("""
        SELECT analyst_email, daily_alpha
        FROM PeerAnalystAlphaDaily
        WHERE trade_date <= ?
    """, conn, params=(trade_date,))
    
    ir_metrics = []
    if not history_df.empty:
        for email, group in history_df.groupby('analyst_email'):
            if len(group) >= 20:
                avg = group['daily_alpha'].mean()
                std = group['daily_alpha'].std()
                ir = avg / std if std > 0 else 0
            else:
                ir = None
                
            ir_metrics.append({
                'email_key': email.lower().strip(),
                'information_ratio': ir
            })
            
    if ir_metrics:
        ir_df = pd.DataFrame(ir_metrics)
    else:
        ir_df = pd.DataFrame(columns=['email_key', 'information_ratio'])
        
    conn.close()
    
    # Merge
    df = df.merge(metrics_df, on='email_key', how='left')
    df = df.merge(ir_df, on='email_key', how='left')
    
    # Fill defaults
    df['coverage'] = df['coverage'].fillna(0)
    
    if 'email_key' in df.columns:
        df = df.drop(columns=['email_key'])
        
    # Derived
    df['total_alpha'] = (df['index_value'] - 1) * 100
    df['hit_rate'] = (df['cumulative_hits'] / df['cumulative_total'] * 100).fillna(0)
    
    df = df.sort_values('index_value', ascending=False)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def render_analyst_alpha_tab(scorecard: pd.DataFrame, meta: dict, use_peer: bool = False):
    """Render content for a specific Analyst Alpha tab."""
    if scorecard.empty:
        st.warning("No data available.")
        return
        
    # --- GLOBAL DATE SELECTOR FOR TAB ---
    # Determine available dates based on data source
    # We use a representative analyst (first one) or specific function to get all dates?
    # Actually, we can just use the dates from the scorecard's latest update or query one analyst.
    # Let's use get_trading_dates from database if possible, or just use meta['end_date'] as default.
    # A safe bet: Get dates for the first analyst found. Most analysts share same trading days.
    
    first_email = scorecard.iloc[0]['analyst_email']
    if use_peer:
        available_dates = get_available_dates_peer(first_email)
    else:
        available_dates = get_available_dates(first_email)
        
    if not available_dates:
        st.warning("No dates available.")
        return

    # Date Dropdown at the top
    selected_date = st.selectbox(
        "📅 Select Date (for Rankings & Details)", 
        available_dates,
        index=0,
        key=f"global_date_selector_{'peer' if use_peer else 'vnindex'}"
    )
    
    # --- LOAD DYNAMIC SCORECARD BASED ON DATE ---
    # Map Names from original scorecard
    name_map = dict(zip(scorecard['analyst_email'], scorecard['analyst_name']))
    coverage_map = dict(zip(scorecard['analyst_email'], scorecard['coverage'])) if 'coverage' in scorecard.columns else {}
    conviction_map = dict(zip(scorecard['analyst_email'], scorecard['conviction'])) if 'conviction' in scorecard.columns else {}
    ir_map = dict(zip(scorecard['analyst_email'], scorecard['information_ratio'])) if 'information_ratio' in scorecard.columns else {}
    
    # If selected date is NOT latest, load historical
    is_latest = (selected_date == available_dates[0])
    
    if is_latest:
        current_scorecard = scorecard.copy()
    else:
        if use_peer:
            current_scorecard = load_peer_daily_scorecard(selected_date)
        else:
            current_scorecard = load_daily_scorecard(selected_date)
            
        # Map names back
        current_scorecard['analyst_name'] = current_scorecard['analyst_email'].map(name_map)
        
    st.markdown("---")
        
    # View selector: Team Average or Individual Analyst
    analyst_options = scorecard[['analyst_name', 'analyst_email']].values.tolist()
    view_options = ['Team Average'] + [a[0] for a in analyst_options]
    
    selected_view = st.selectbox(
        "Select View",
        view_options,
        index=0,
        key=f"analyst_selector_{'peer' if use_peer else 'vnindex'}"
    )
    
    st.markdown("---")
    
    # Display chart based on selection
    if selected_view == 'Team Average':
        # Team Average Chart (Always shows full history, independent of selected date usually, 
        # BUT user might expect it to cut off? Usually Chart shows TREND, so full history is better.)
        if meta:
            # Load all analysts' histories and calculate team average
            all_histories = []
            for _, row in scorecard.iterrows():
                if use_peer:
                    history = load_peer_analyst_history(row['analyst_email'])
                else:
                    history = load_analyst_history(row['analyst_email'])
                    
                if not history.empty:
                    history = history.set_index('date')['index_value']
                    all_histories.append(history)
            
            if all_histories:
                team_df = pd.DataFrame(all_histories).T
                team_df['team_avg'] = team_df.mean(axis=1)
                team_df = team_df.reset_index()
                team_df = team_df.rename(columns={'index': 'date'})
                
                # Load VnIndex for comparison (only for vs VnIndex tab)
                if not use_peer:
                    vnindex = load_vnindex_normalized(meta['start_date'], meta['end_date'])
                    if not vnindex.empty:
                        team_df = team_df.merge(
                            vnindex.rename(columns={'index_value': 'vnindex_normalized'}),
                            on='date', how='left'
                        )
                
                fig = create_team_overview_chart(team_df, benchmark_label="Sector Avg = 0%" if use_peer else None)
                # Add vertical line for selected date? Optional.
                st.plotly_chart(fig, use_container_width=True)
    else:
        # Individual Analyst Chart
        selected_name = selected_view
        selected_email = next((email for name, email in analyst_options if name == selected_name), None)
        
        if selected_email:
            if use_peer:
                history = load_peer_analyst_history(selected_email)
                if not history.empty:
                    fig = create_peer_alpha_chart(history, title=f"Peer Alpha - {selected_name}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                history = load_analyst_history(selected_email)
                if not history.empty:
                    vnindex = load_vnindex_normalized(meta['start_date'], meta['end_date'])
                    if not vnindex.empty:
                        fig = create_alpha_vs_vnindex_chart(history, vnindex, title=f"Alpha - {selected_name}")
                    else:
                        fig = create_alpha_vs_vnindex_chart(history, pd.DataFrame(), title=f"Alpha - {selected_name}")
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # Summary / Ranking Table (Requested Feature)
    st.subheader(f"📊 Ranking Overview ({'vs Sector Avg' if use_peer else 'vs VnIndex'})")
    if not is_latest:
        st.caption(f"📅 Showing data as of: **{selected_date}**")
    
    # Format and Display dynamic scorecard
    display_df = current_scorecard.copy()
    
    display_cols = ['rank', 'analyst_name', 'total_alpha', 'hit_rate', 
                    'information_ratio', 'conviction', 'coverage']
    
    # Filter columns that exist
    display_cols = [c for c in display_cols if c in display_df.columns]
    
    summary_table = display_df[display_cols].copy()
    
    # Explicit renaming for better clarity
    rename_map = {
        'rank': 'Rank',
        'analyst_name': 'Analyst Name',
        'total_alpha': 'Total Alpha',
        'ytd_alpha': 'Total Alpha', # Handle legacy name if present
        'hit_rate': 'Hit Rate',
        'information_ratio': 'Information Ratio',
        'conviction': 'Conviction',
        'coverage': 'Coverage'
    }
    summary_table = summary_table.rename(columns=rename_map)
    # Fallback for others
    summary_table.columns = [c.replace('_', ' ').title() if c not in rename_map.values() else c for c in summary_table.columns]
    
    # Check nulls for hist data
    summary_table = summary_table.fillna("N/A")

    # Format
    if 'Total Alpha' in summary_table.columns:
        summary_table['Total Alpha'] = summary_table['Total Alpha'].apply(lambda x: f"{float(x):+.2f}%" if x != "N/A" and pd.notna(x) else "N/A")
    if 'Hit Rate' in summary_table.columns:
        summary_table['Hit Rate'] = summary_table['Hit Rate'].apply(lambda x: f"{float(x):.1f}%" if x != "N/A" and pd.notna(x) else "N/A")
    if 'Information Ratio' in summary_table.columns:
        summary_table['Information Ratio'] = summary_table['Information Ratio'].apply(lambda x: f"{float(x):.4f}" if x != "N/A" and pd.notna(x) else "N/A")
    if 'Conviction' in summary_table.columns:
        summary_table['Conviction'] = summary_table['Conviction'].apply(lambda x: f"{float(x):.1f}%" if x != "N/A" and pd.notna(x) else "N/A")
        
    st.dataframe(summary_table, use_container_width=True, hide_index=True)
    
    # DEBUG: Check why columns might be N/A
    # st.write("DEBUG: Raw Joined Data Sample", display_df.head(3))
    # st.write("DEBUG: Selected Date", selected_date)
    # st.write(f"DEBUG: Has Coverage? {'coverage' in display_df.columns}")
    # st.write(f"DEBUG: Has Conviction? {'conviction' in display_df.columns}")

    st.markdown("---")

    # Detail Table (Only if specific analyst selected)
    if selected_view != 'Team Average' and selected_email:
        st.subheader(f"📋 Daily Details - {selected_name}")
        
        # Load daily contributions for SELECTED DATE
        if use_peer:
            contributions = load_peer_daily_contributions(selected_email, selected_date)
            benchmark_col = "peer_return"
            benchmark_label = "Sector Avg"
        else:
            contributions = load_daily_contributions(selected_email, selected_date)
            benchmark_col = "vnindex_return"
            benchmark_label = "VnIndex"
            
        if not contributions.empty:
            # Render contributions table
            display_df = contributions.copy()
            rename_dict = {
                'ticker': 'Ticker',
                'recommendation': 'Rating',
                'direction_weight': 'Direction',
                'stock_return': 'Stock %',
                benchmark_col: f'{benchmark_label} %',
                'excess_return': 'Excess %',
                'contribution': 'Contrib %',
                'is_correct': '✓'
            }
            display_df = display_df.rename(columns=rename_dict)
            
            # Format
            display_df['Stock %'] = display_df['Stock %'].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
            display_df[f'{benchmark_label} %'] = display_df[f'{benchmark_label} %'].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
            display_df['Excess %'] = display_df['Excess %'].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
            display_df['Contrib %'] = display_df['Contrib %'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "")
            display_df['Direction'] = display_df['Direction'].apply(lambda x: f"{x:+.1f}")
            display_df['✓'] = display_df['✓'].apply(lambda x: "✅" if x == 1 else "❌")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
        else:
            st.info(f"No transactions for {selected_date}")


def render_performance_overview(scorecard: pd.DataFrame, meta: dict):
    """Render Performance Overview page (Team + Analyst combined)."""
    st.subheader("Performance Overview")
    
    # View selector: Team Average or Individual Analyst
    analyst_options = scorecard[['analyst_name', 'analyst_email']].values.tolist()
    view_options = ['Team Average'] + [a[0] for a in analyst_options]
    
    selected_view = st.selectbox(
        "Select View",
        view_options,
        index=0
    )
    
    st.markdown("---")
    
    # Display chart based on selection
    if selected_view == 'Team Average':
        # Team Average Chart
        if meta:
            # Load all analysts' histories and calculate team average
            all_histories = []
            for _, row in scorecard.iterrows():
                history = load_analyst_history(row['analyst_email'])
                if not history.empty:
                    history = history.set_index('date')['index_value']
                    all_histories.append(history)
            
            if all_histories:
                team_df = pd.DataFrame(all_histories).T
                team_df['team_avg'] = team_df.mean(axis=1)
                team_df = team_df.reset_index()
                team_df = team_df.rename(columns={'index': 'date'})
                
                # Load VnIndex
                vnindex = load_vnindex_normalized(meta['start_date'], meta['end_date'])
                if not vnindex.empty:
                    # Merge VnIndex
                    team_df = team_df.merge(
                        vnindex.rename(columns={'index_value': 'vnindex_normalized'}),
                        on='date', how='left'
                    )
                
                fig = create_team_overview_chart(team_df)
                st.plotly_chart(fig, use_container_width=True)
    else:
        # Individual Analyst Chart
        selected_name = selected_view
        
        # Get analyst email
        selected_email = None
        for name, email in analyst_options:
            if name == selected_name:
                selected_email = email
                break
        
        # Load history (cached per analyst)
        alpha_history = load_analyst_history(selected_email)
        
        # Alpha Index History Chart
        if not alpha_history.empty and meta:
            # Load active tickers and merge
            ticker_list = load_analyst_ticker_list(selected_email)
            if not ticker_list.empty:
                alpha_history = alpha_history.merge(ticker_list, on='date', how='left')
            
            vnindex = load_vnindex_normalized(meta['start_date'], meta['end_date'])
            
            if not vnindex.empty:
                fig = create_alpha_vs_vnindex_chart(
                    alpha_history, vnindex,
                    title=f"Alpha Index History - {selected_name}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Analyst Scorecard Table
    st.subheader("Analyst Scorecard")
    
    display_cols = ['rank', 'analyst_name', 'index_value', 'ytd_alpha', 'hit_rate', 
                    'information_ratio', 'conviction', 'coverage']
    display_df = scorecard[display_cols].copy()
    display_df.columns = ['Rank', 'Analyst', 'Alpha Index', 'YTD Alpha %', 'Hit Rate %',
                          'Info Ratio', 'Conviction %', 'Coverage']
    
    display_df['Alpha Index'] = display_df['Alpha Index'].apply(lambda x: f"{x:.2f}")
    display_df['YTD Alpha %'] = display_df['YTD Alpha %'].apply(lambda x: f"{x:+.2f}")
    display_df['Hit Rate %'] = display_df['Hit Rate %'].apply(lambda x: f"{x:.1f}")
    display_df['Info Ratio'] = display_df['Info Ratio'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display_df['Conviction %'] = display_df['Conviction %'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_peer_comparison(peer_scorecard: pd.DataFrame, meta: dict):
    """Render Peer Comparison page - using average peer return as benchmark."""
    st.subheader("🔄 Peer Comparison")
    
    st.info("""
    **Peer Comparison** so sánh hiệu suất của mỗi cổ phiếu với **trung bình return của tất cả cổ phiếu trong list cover** 
    (thay vì so với VnIndex như trang Performance Overview).
    
    - **Benchmark**: Avg Return của list cover (equal-weighted)
    - **Excess Return** = Stock Return - Avg Peer Return
    """)
    
    # View selector: Team Average or Individual Analyst
    analyst_options = peer_scorecard[['analyst_name', 'analyst_email']].values.tolist()
    view_options = ['Team Average'] + [a[0] for a in analyst_options]
    
    selected_view = st.selectbox(
        "Select View",
        view_options,
        index=0,
        key="peer_view_selector"
    )
    
    st.markdown("---")
    
    # Display chart based on selection
    if selected_view == 'Team Average':
        # Team Average Chart - using peer benchmark
        if meta:
            # Load all analysts' peer histories and calculate team average
            all_histories = []
            for _, row in peer_scorecard.iterrows():
                history = load_peer_analyst_history(row['analyst_email'])
                if not history.empty:
                    history = history.set_index('date')['index_value']
                    all_histories.append(history)
            
            if all_histories:
                team_df = pd.DataFrame(all_histories).T
                team_df['team_avg'] = team_df.mean(axis=1)
                team_df = team_df.reset_index()
                team_df = team_df.rename(columns={'index': 'date'})
                
                # Create chart showing team average peer-based index
                fig = create_team_overview_chart(team_df, benchmark_label="Peer Benchmark = 1.0")
                st.plotly_chart(fig, use_container_width=True)
    else:
        # Individual Analyst Chart using peer benchmark
        selected_name = selected_view
        
        # Get analyst email
        selected_email = None
        for name, email in analyst_options:
            if name == selected_name:
                selected_email = email
                break
        
        # Load peer-based history
        alpha_history = load_peer_analyst_history(selected_email)
        
        # Alpha Index History Chart (peer-based)
        if not alpha_history.empty and meta:
            # Load active tickers and merge
            ticker_list = load_analyst_ticker_list(selected_email)
            if not ticker_list.empty:
                alpha_history = alpha_history.merge(ticker_list, on='date', how='left')
            
            # Create chart with peer benchmark line at 1.0
            from components.charts import create_peer_alpha_chart
            fig = create_peer_alpha_chart(
                alpha_history,
                title=f"Alpha Index vs Coverage - {selected_name}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Peer Scorecard Table
    st.subheader("Analyst Scorecard (Peer Benchmark)")
    
    display_cols = ['rank', 'analyst_name', 'index_value', 'ytd_alpha', 'hit_rate', 
                    'information_ratio', 'conviction', 'coverage']
    display_df = peer_scorecard[display_cols].copy()
    display_df.columns = ['Rank', 'Analyst', 'Peer Alpha Index', 'YTD Alpha %', 'Hit Rate %',
                          'Info Ratio', 'Conviction %', 'Coverage']
    
    display_df['Peer Alpha Index'] = display_df['Peer Alpha Index'].apply(lambda x: f"{x:.2f}")
    display_df['YTD Alpha %'] = display_df['YTD Alpha %'].apply(lambda x: f"{x:+.2f}")
    display_df['Hit Rate %'] = display_df['Hit Rate %'].apply(lambda x: f"{x:.1f}")
    display_df['Info Ratio'] = display_df['Info Ratio'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display_df['Conviction %'] = display_df['Conviction %'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_industry_alpha(meta: dict):
    """Render Industry Alpha page - Alpha by Industry."""
    st.subheader("🏭 Industry Alpha")
    
    st.info("""
    **Industry Alpha** tính toán hiệu suất theo **Industry** thay vì theo Analyst.
    Tất cả cổ phiếu trong cùng một ngành được gom nhóm lại để tính Alpha Index.
    """)
    
    # Load industry scorecards
    industry_scorecard = load_industry_scorecard()
    industry_peer_scorecard = load_industry_peer_scorecard()
    
    if industry_scorecard.empty:
        st.warning("⚠️ Industry Alpha data not found. Please run `python precalculate.py` to generate data.")
        return
    
    # Tabs for benchmark selection
    tab1, tab2 = st.tabs(["📊 vs VnIndex", "🔄 vs Sector Average"])
    
    with tab1:
        render_industry_alpha_tab(industry_scorecard, meta, use_peer=False)
    
    with tab2:
        render_industry_alpha_tab(industry_peer_scorecard, meta, use_peer=True)


def render_industry_alpha_tab(scorecard: pd.DataFrame, meta: dict, use_peer: bool = False):
    """Render Industry Alpha tab content."""
    if scorecard.empty:
        st.warning("No data available.")
        return
    
    benchmark_label = "Sector Average" if use_peer else "VnIndex"
    
    # Industry selector
    industries = scorecard['industry'].tolist()
    selected_industry = st.selectbox(
        "Select Industry",
        industries,
        key=f"industry_selector_{'peer' if use_peer else 'vnindex'}"
    )
    
    st.markdown("---")
    
    # Load and display chart
    if use_peer:
        alpha_history = load_industry_peer_history(selected_industry)
    else:
        alpha_history = load_industry_history(selected_industry)
    
    if not alpha_history.empty and meta:
        from components.charts import create_peer_alpha_chart
        
        if use_peer:
            fig = create_peer_alpha_chart(
                alpha_history,
                title=f"Industry Alpha - {selected_industry} (vs Sector Avg)"
            )
        else:
            # Load VnIndex for comparison
            vnindex = load_vnindex_normalized(meta['start_date'], meta['end_date'])
            if not vnindex.empty:
                fig = create_alpha_vs_vnindex_chart(
                    alpha_history, vnindex,
                    title=f"Industry Alpha - {selected_industry} (vs VnIndex)"
                )
            else:
                fig = create_peer_alpha_chart(
                    alpha_history,
                    title=f"Industry Alpha - {selected_industry}"
                )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Date selector
    available_dates = get_industry_available_dates(selected_industry, use_peer)
    if not available_dates:
        st.info("No stock detail data available.")
        return
    
    selected_date = st.selectbox(
        "Select Date",
        available_dates,
        key=f"industry_date_{'peer' if use_peer else 'vnindex'}"
    )

    st.markdown("---")
    
    # Global Industry Ranking Table (Dynamic for Selected Date)
    st.subheader(f"📊 Industry Ranking Overview ({pd.to_datetime(selected_date).strftime('%Y-%m-%d')})")
    
    # We need to calculate this dynamically for the selected date
    # 1. Get all industries
    # 2. For each industry, get its Alpha Index on that date
    # 3. Count active stocks on that date
    
    ranking_data = []
    
    # Use cached data if possible, but simplest is to iterate
    # Note: efficient way is to load IndustryAlphaDaily for all industries on this date
    # But we don't have a single function for that yet. 
    # Let's iterate over 'industries' list from scorecard (which has all industries)
    
    for ind in industries:
        # Load alpha history for this industry to get index value on date
        # Optimziation: Create a bulk loader function later if slow.
        if use_peer:
             # This might be slow if we load history 18 times on every refresh.
             # Better approach: Add a new SQL query in database.py or use IndustryDailyContributions
             pass
    
    # ALTERNATIVE FAST APPROACH: 
    # Use `get_industry_ranking_on_date(date)` function which we should create.
    # Since we can't easily create new SQL function without modifying precalculate/database deeply?
    # Actually we can add query here or in app.py helper.
    
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    if use_peer:
        # Optimziation: Total count is already saved in IndustryPeerAlphaDaily
        query_ranking = """
            SELECT industry, index_value, total as stock_count
            FROM IndustryPeerAlphaDaily
            WHERE trade_date = ?
        """
    else:
        query_ranking = """
            SELECT industry, index_value, total as stock_count
            FROM IndustryAlphaDaily
            WHERE trade_date = ?
        """
    
    df_summary = pd.read_sql_query(query_ranking, conn, params=(selected_date,))
    conn.close()
    
    # Process
    if not df_summary.empty:
        # Rename
        df_summary = df_summary.rename(columns={
            'industry': 'Industry',
            'index_value': 'Alpha Index',
            'stock_count': 'Stocks'
        })
        
        # Sort
        df_summary = df_summary.sort_values(by='Alpha Index', ascending=False)
        
        # Format
        df_summary['Alpha Index'] = df_summary['Alpha Index'].apply(lambda x: f"{x:.2f}")
        df_summary['Stocks'] = df_summary['Stocks'].astype(int)
        
        # Display
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
    else:
        st.info(f"No summary data for {selected_date}")

    
    st.markdown("---")
    
    # Stock Detail Table
    st.subheader(f"📋 Stock Details - {selected_industry}")
    
    # Load contributions
    if use_peer:
        contributions = load_industry_peer_contributions(selected_industry, selected_date)
        benchmark_col = "sector_return"
    else:
        contributions = load_industry_daily_contributions(selected_industry, selected_date)
        benchmark_col = "vnindex_return"
    
    if contributions.empty:
        st.info(f"No data for {selected_date}")
        return
    
    display_df = contributions.copy()
    display_df = display_df.drop(columns=['trade_date'])
    
    col_rename = {
        'ticker': 'Ticker',
        'recommendation': 'Rating',
        'direction_weight': 'Direction',
        'stock_return': 'Stock %',
        benchmark_col: f'{benchmark_label} %',
        'excess_return': 'Excess %',
        'contribution': 'Contrib %',
        'is_correct': '✓'
    }
    display_df = display_df.rename(columns=col_rename)
    
    display_df['Stock %'] = display_df['Stock %'].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")
    display_df[f'{benchmark_label} %'] = display_df[f'{benchmark_label} %'].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")
    display_df['Excess %'] = display_df['Excess %'].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")
    display_df['Contrib %'] = display_df['Contrib %'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "N/A")
    display_df['Direction'] = display_df['Direction'].apply(lambda x: f"{x:+.1f}")
    display_df['✓'] = display_df['✓'].apply(lambda x: "✅" if x == 1 else "❌")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_leaderboard(scorecard: pd.DataFrame):
    """Render Leaderboard page."""
    st.subheader("🏆 Analyst Leaderboard")
    
    # Top 3 Awards
    col1, col2, col3 = st.columns(3)
    
    if len(scorecard) >= 1:
        with col1:
            top_alpha = scorecard.iloc[0]
            st.markdown("### 🥇 Alpha Champion")
            st.markdown(f"**{top_alpha['analyst_name']}**")
            st.markdown(f"Total Alpha: {top_alpha['ytd_alpha']:+.2f}%")
    
    if len(scorecard) >= 2:
        best_hit = scorecard.loc[scorecard['hit_rate'].idxmax()]
        with col2:
            st.markdown("### 🎯 Accuracy Leader")
            st.markdown(f"**{best_hit['analyst_name']}**")
            st.markdown(f"Hit Rate: {best_hit['hit_rate']:.1f}%")
    
    if len(scorecard) >= 3:
        ir_df = scorecard[scorecard['information_ratio'].notna()]
        if not ir_df.empty:
            best_ir = ir_df.loc[ir_df['information_ratio'].idxmax()]
            with col3:
                st.markdown("### 📊 Consistency King")
                st.markdown(f"**{best_ir['analyst_name']}**")
                st.markdown(f"IR: {best_ir['information_ratio']:.2f}")
    
    st.markdown("---")
    
    # Ranking Chart
    tab1, tab2, tab3 = st.tabs(["By Total Alpha", "By Hit Rate", "By Info Ratio"])
    
    with tab1:
        fig = create_ranking_bars(scorecard, 'ytd_alpha', 'Rankings by Total Alpha')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = create_ranking_bars(scorecard, 'hit_rate', 'Rankings by Hit Rate')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        ir_df = scorecard[scorecard['information_ratio'].notna()].copy()
        if not ir_df.empty:
            fig = create_ranking_bars(ir_df, 'information_ratio', 'Rankings by Information Ratio')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for Information Ratio")


def render_calculation_detail(scorecard: pd.DataFrame, peer_scorecard: pd.DataFrame = None):
    """Render Calculation Detail page for verifying data."""
    st.subheader("📋 Calculation Detail - Data Verification")
    
    st.markdown("""
    Bảng này hiển thị **chi tiết tính toán Alpha Index** để bạn có thể kiểm tra:
    - **Stock Return**: Lợi nhuận cổ phiếu (%)
    - **VnIndex Return**: Lợi nhuận VnIndex (%)
    - **Excess Return**: Stock Return - VnIndex Return
    - **Direction Weight**: +1.0 (OPF/BUY), -1.0 (UPF/SELL/AVOID), +0.3 (MPF)
    - **Contribution**: Direction × Excess Return
    - **Daily Alpha**: Trung bình của tất cả Contributions
    """)
    
    st.markdown("---")
    
    # Analyst selector
    analyst_options = scorecard[['analyst_name', 'analyst_email']].values.tolist()
    selected_name = st.selectbox(
        "Select Analyst", 
        [a[0] for a in analyst_options],
        key="calc_detail_analyst"
    )
    
    # Get analyst email
    selected_email = None
    for name, email in analyst_options:
        if name == selected_name:
            selected_email = email
            break
    
    # Benchmark selector
    col1, col2 = st.columns([1, 3])
    with col1:
        benchmark_type = st.selectbox(
            "Benchmark Type",
            ["VnIndex Benchmark", "Peer Benchmark"],
            key="benchmark_selector"
        )
    
    # Get available dates based on benchmark type
    if benchmark_type == "Peer Benchmark":
        available_dates = get_peer_available_dates(selected_email)
    else:
        available_dates = get_available_dates(selected_email)
    
    if not available_dates:
        st.warning("No calculation data available. Please run `python precalculate.py` first.")
        return
    
    # Date selector
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_date = st.selectbox(
            "Select Date",
            available_dates,
            key="calc_detail_date"
        )
    
    # Load contributions for selected date based on benchmark type
    if benchmark_type == "Peer Benchmark":
        contributions = load_peer_daily_contributions(selected_email, selected_date)
        benchmark_label = "Peer Return"
        benchmark_col = "peer_return"
    else:
        contributions = load_daily_contributions(selected_email, selected_date)
        benchmark_label = "VnIndex Return"
        benchmark_col = "vnindex_return"
    
    if contributions.empty:
        st.info(f"No data for {selected_date}")
        return
    
    # Show summary for this date
    st.markdown(f"### 📅 {selected_date}")
    
    total_stocks = len(contributions)
    correct_calls = contributions['is_correct'].sum()
    daily_alpha = contributions['contribution'].mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stocks Rated", total_stocks)
    with col2:
        st.metric("Correct Calls", f"{correct_calls}/{total_stocks}", 
                  f"{correct_calls/total_stocks*100:.1f}%")
    with col3:
        st.metric("Daily Alpha", f"{daily_alpha:+.4f}%")
    
    st.markdown("---")
    
    # Format and display detailed table
    display_df = contributions.copy()
    
    # Rename columns for display - dynamic based on benchmark type
    rename_dict = {
        'ticker': 'Ticker',
        'recommendation': 'Recommendation',
        'direction_weight': 'Direction',
        'stock_return': 'Stock Return %',
        benchmark_col: f'{benchmark_label} %',
        'excess_return': 'Excess Return %',
        'contribution': 'Contribution %',
        'is_correct': 'Correct'
    }
    display_df = display_df.rename(columns=rename_dict)
    
    # Drop trade_date column (already shown above)
    display_df = display_df.drop(columns=['trade_date'])
    
    # Format numbers
    display_df['Stock Return %'] = display_df['Stock Return %'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "N/A")
    display_df[f'{benchmark_label} %'] = display_df[f'{benchmark_label} %'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "N/A")
    display_df['Excess Return %'] = display_df['Excess Return %'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "N/A")
    display_df['Contribution %'] = display_df['Contribution %'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "N/A")
    display_df['Direction'] = display_df['Direction'].apply(lambda x: f"{x:+.1f}")
    display_df['Correct'] = display_df['Correct'].apply(lambda x: "✅" if x == 1 else "❌")
    
    st.subheader("Stock-Level Contributions")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Formula explanation
    with st.expander("📐 Calculation Formula - VnIndex Benchmark"):
        st.markdown("""
        ### How Daily Alpha is calculated (Performance Overview):
        
        1. **Excess Return** = Stock Return - VnIndex Return
        2. **Contribution** = Direction Weight × Excess Return
        3. **Daily Alpha** = Average of all Contributions
        4. **Index Update** = Previous Index × (1 + Daily Alpha / 100)
        
        ### 🔄 Stock Transfer Logic (Current Owner):
        - A stock belongs to **only one analyst** at a time (the "Current Owner").
        - **Current Owner** is defined as the analyst with the **most recent** recommendation.
        - When a new analyst issues a report for a stock:
            - Ownership **transfers immediately** to the new analyst.
            - The stock **stops contributing** to the previous analyst's performance.
        
        ### Direction Weights:
        - **OPF (Outperform, BUY)**: +1.0 → Bullish, positive contribution when stock beats index
        - **UPF (Underperform, SELL, AVOID)**: -1.0 → Bearish, positive contribution when stock trails index
        - **MPF (Market-Perform)**: +0.3 → Light bullish, small positive when stock beats index
        - **WATCH**: 0 → Excluded from calculation
        
        ### Correct Call:
        - **OPF**: Correct if Excess Return > 0
        - **UPF/MPF**: Correct if Excess Return < 0
        """)
    
    # Peer Comparison explanation
    with st.expander("🔄 Calculation Formula - Peer Comparison"):
        st.markdown("""
        ### How Daily Alpha is calculated (Peer Comparison):
        
        Trang **Peer Comparison** sử dụng cách tính tương tự nhưng thay **VnIndex Return** bằng 
        **Average Return của tất cả cổ phiếu trong list cover**.
        
        | Metric | Performance Overview | Peer Comparison |
        |--------|---------------------|-----------------|
        | **Benchmark** | VnIndex Return | Avg Return của list cover |
        | **Excess Return** | Stock Return - VnIndex Return | Stock Return - Avg Peer Return |
        
        ### 📊 Cách tính Avg Peer Return:
        
        1. **Cho mỗi analyst**: Lấy tất cả cổ phiếu active của analyst ngày đó
        2. Tính **equal-weighted average return** của tất cả cổ phiếu đó
        3. **Excess Return** = Stock Return - Avg Peer Return
        
        ### 🎯 Ý nghĩa:
        
        - **Performance Overview (vs VnIndex)**: Đánh giá khả năng chọn stock outperform/underperform thị trường
        - **Peer Comparison (vs Peers)**: Đánh giá khả năng **chọn stock tốt nhất** trong list cover của analyst
        
        ### Ví dụ:
        
        Nếu ngày X analyst có list cover:
        - VNM: +2%
        - HPG: +3%  
        - FPT: +1%
        - VHM: +4%
        
        **Avg Peer Return** = (2+3+1+4)/4 = **2.5%**
        
        | Stock | Return | Excess vs VnIndex (giả sử +1%) | Excess vs Peers (2.5%) |
        |-------|--------|-------------------------------|------------------------|
        | VNM | +2% | +2% - 1% = **+1%** | +2% - 2.5% = **-0.5%** |
        | HPG | +3% | +3% - 1% = **+2%** | +3% - 2.5% = **+0.5%** |
        | VHM | +4% | +4% - 1% = **+3%** | +4% - 2.5% = **+1.5%** |
        
        → Peer Comparison đánh giá **relative ranking** trong universe của analyst.
        """)


if __name__ == "__main__":
    main()
