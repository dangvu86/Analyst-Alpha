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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import COLORS, VNINDEX_TICKER
from components.charts import (
    create_alpha_vs_vnindex_chart, create_daily_alpha_bars,
    create_ranking_bars, create_team_overview_chart
)
from data.database import get_vnindex_prices

# Pre-calculated database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALPHA_INDEX_DB = os.path.join(BASE_DIR, 'alpha_index.db')


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


@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
def load_analyst_history(analyst_email: str) -> pd.DataFrame:
    """Load alpha history for a specific analyst."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    df = pd.read_sql_query("""
        SELECT 
            trade_date as date,
            daily_alpha,
            index_value,
            hits,
            total,
            cumulative_hits,
            cumulative_total
        FROM AnalystAlphaDaily
        WHERE analyst_email = ?
        ORDER BY trade_date
    """, conn, params=(analyst_email,))
    
    conn.close()
    return df


@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
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


# ===== PEER COMPARISON DATA LOADING FUNCTIONS =====

@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
def load_peer_analyst_history(analyst_email: str) -> pd.DataFrame:
    """Load peer-based alpha history for a specific analyst."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    df = pd.read_sql_query("""
        SELECT 
            trade_date as date,
            daily_alpha,
            index_value,
            hits,
            total,
            cumulative_hits,
            cumulative_total
        FROM PeerAnalystAlphaDaily
        WHERE analyst_email = ?
        ORDER BY trade_date
    """, conn, params=(analyst_email,))
    
    conn.close()
    return df


@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
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

@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
def load_industry_history(industry: str) -> pd.DataFrame:
    """Load alpha history for a specific industry (vs VnIndex)."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    df = pd.read_sql_query("""
        SELECT trade_date as date, daily_alpha, index_value, hits, total,
               cumulative_hits, cumulative_total
        FROM IndustryAlphaDaily WHERE industry = ? ORDER BY trade_date
    """, conn, params=(industry,))
    conn.close()
    return df


@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
def load_industry_peer_history(industry: str) -> pd.DataFrame:
    """Load alpha history for an industry (vs Sector Average)."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    df = pd.read_sql_query("""
        SELECT trade_date as date, daily_alpha, index_value, hits, total,
               cumulative_hits, cumulative_total
        FROM IndustryPeerAlphaDaily WHERE industry = ? ORDER BY trade_date
    """, conn, params=(industry,))
    conn.close()
    return df


@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=3600)
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
        ["📊 Performance Overview", "🔄 Peer Comparison", "🏭 Industry Alpha", "📋 Calculation Detail", "🏆 Leaderboard"]
    )
    
    if page == "📊 Performance Overview":
        render_performance_overview(scorecard, meta)
    elif page == "🔄 Peer Comparison":
        render_peer_comparison(peer_scorecard, meta)
    elif page == "🏭 Industry Alpha":
        render_industry_alpha(meta)
    elif page == "📋 Calculation Detail":
        render_calculation_detail(scorecard, peer_scorecard)
    elif page == "🏆 Leaderboard":
        render_leaderboard(scorecard)


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
    
    # Stock Detail Table
    st.subheader(f"📋 Stock Details - {selected_industry}")
    
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
