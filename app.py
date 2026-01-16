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
    
    if scorecard.empty:
        st.warning("No analyst data available.")
        return
    
    # Navigation
    st.sidebar.title("⚙️ Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Team Overview", "👤 Analyst Detail", "📋 Calculation Detail", "🏆 Leaderboard"]
    )
    
    if page == "🏠 Team Overview":
        render_team_overview(scorecard, meta)
    elif page == "👤 Analyst Detail":
        render_analyst_detail(scorecard, meta)
    elif page == "📋 Calculation Detail":
        render_calculation_detail(scorecard)
    elif page == "🏆 Leaderboard":
        render_leaderboard(scorecard)


def render_team_overview(scorecard: pd.DataFrame, meta: dict):
    """Render Team Overview page."""
    st.subheader("Team Performance Overview")
    
    # KPI Cards
    team_avg_alpha = scorecard['index_value'].mean()
    best_performer = scorecard.iloc[0]['analyst_name'] if not scorecard.empty else "N/A"
    best_alpha = scorecard.iloc[0]['ytd_alpha'] if not scorecard.empty else 0
    above_100 = len(scorecard[scorecard['index_value'] > 1])
    total_analysts = len(scorecard)
    avg_hit_rate = scorecard['hit_rate'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Team Avg Alpha", f"{team_avg_alpha:.4f}", f"{(team_avg_alpha - 1) * 100:+.2f}%")
    with col2:
        st.metric("Best Performer", best_performer, f"{best_alpha:+.2f}%")
    with col3:
        st.metric("Above 1.0", f"{above_100}/{total_analysts}", f"{above_100/total_analysts*100:.0f}%")
    with col4:
        st.metric("Avg Hit Rate", f"{avg_hit_rate:.1f}%")
    
    st.markdown("---")
    
    # Team Alpha Chart
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
    
    # Methodology explanation
    with st.expander("📐 Cách tính Team Index và VnIndex"):
        st.markdown("""
        ### 📊 Team Average (Đường xanh dương)
        
        **Công thức:**
        ```
        Team Average = (Index₁ + Index₂ + ... + Index₁₅) / 15
        ```
        
        Với mỗi ngày giao dịch, Team Average là **trung bình cộng đơn giản** của Alpha Index 
        của tất cả 15 analysts vào ngày đó.
        
        ---
        
        ### 📈 VnIndex (Đường ghi đậm)
        
        **Cách normalize:**
        ```
        Normalized VnIndex = (Giá VnIndex ngày hiện tại / Giá VnIndex ngày đầu tiên) × 100
        ```
        
        - VnIndex được **normalized về 100** tại điểm bắt đầu (2020-01-01)
        - Điều này cho phép so sánh trực tiếp với Alpha Index (cũng bắt đầu từ 100)
        
        **Ví dụ:**
        - Nếu VnIndex ngày 1/1/2020 = 960
        - VnIndex ngày 1/1/2026 = 1250
        - Normalized = 1250 / 960 × 100 = **130.2**
        
        ---
        
        ### ✅ Diễn giải
        
        - **Team Average > VnIndex**: Team outperform thị trường
        - **Team Average < VnIndex**: Team underperform thị trường
        - **Analyst Index > 100**: Analyst tạo ra alpha dương
        - **Analyst Index < 100**: Analyst tạo ra alpha âm
        """)
    
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


def render_analyst_detail(scorecard: pd.DataFrame, meta: dict):
    """Render Individual Analyst Detail page."""
    st.subheader("Individual Analyst Performance")
    
    # Analyst selector
    analyst_options = scorecard[['analyst_name', 'analyst_email']].values.tolist()
    selected_name = st.selectbox(
        "Select Analyst", 
        [a[0] for a in analyst_options]
    )
    
    # Get analyst email
    selected_email = None
    for name, email in analyst_options:
        if name == selected_name:
            selected_email = email
            break
    
    # Get analyst summary
    analyst_row = scorecard[scorecard['analyst_name'] == selected_name].iloc[0]
    
    # Load history (cached per analyst)
    alpha_history = load_analyst_history(selected_email)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Alpha Index", f"{analyst_row['index_value']:.2f}", 
                  f"{analyst_row['ytd_alpha']:+.2f}%")
    with col2:
        st.metric("Hit Rate", f"{analyst_row['hit_rate']:.1f}%")
    with col3:
        ir = analyst_row['information_ratio']
        st.metric("Info Ratio", f"{ir:.2f}" if pd.notna(ir) else "N/A")
    with col4:
        st.metric("Conviction", f"{analyst_row['conviction']:.1f}%",
                  f"OPF:{analyst_row['opf_count']} UPF:{analyst_row['upf_count']} MPF:{analyst_row['mpf_count']}")
    
    st.markdown("---")
    
    # Alpha Index History Chart
    if not alpha_history.empty and meta:
        vnindex = load_vnindex_normalized(meta['start_date'], meta['end_date'])
        
        if not vnindex.empty:
            fig = create_alpha_vs_vnindex_chart(
                alpha_history, vnindex,
                title=f"Alpha Index History - {selected_name}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Daily Alpha Bar Chart
    col1, col2 = st.columns(2)
    
    with col1:
        if not alpha_history.empty:
            fig = create_daily_alpha_bars(alpha_history, days=30)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Rating Distribution")
        st.write(f"- **OPF (Outperform):** {analyst_row['opf_count']}")
        st.write(f"- **UPF (Underperform):** {analyst_row['upf_count']}")
        st.write(f"- **MPF (Market Perform):** {analyst_row['mpf_count']}")
        st.write(f"- **Total Coverage:** {analyst_row['coverage']}")


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


def render_calculation_detail(scorecard: pd.DataFrame):
    """Render Calculation Detail page for verifying data."""
    st.subheader("📋 Calculation Detail - Data Verification")
    
    st.markdown("""
    Bảng này hiển thị **chi tiết tính toán Alpha Index** để bạn có thể kiểm tra:
    - **Stock Return**: Lợi nhuận cổ phiếu (%)
    - **VnIndex Return**: Lợi nhuận VnIndex (%)
    - **Excess Return**: Stock Return - VnIndex Return
    - **Direction Weight**: +1.0 (OPF), -1.0 (UPF), -0.3 (MPF)
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
    
    # Get available dates
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
    
    # Load contributions for selected date
    contributions = load_daily_contributions(selected_email, selected_date)
    
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
    
    # Rename columns for display
    display_df = display_df.rename(columns={
        'ticker': 'Ticker',
        'recommendation': 'Recommendation',
        'direction_weight': 'Direction',
        'stock_return': 'Stock Return %',
        'vnindex_return': 'VnIndex Return %',
        'excess_return': 'Excess Return %',
        'contribution': 'Contribution %',
        'is_correct': 'Correct'
    })
    
    # Drop trade_date column (already shown above)
    display_df = display_df.drop(columns=['trade_date'])
    
    # Format numbers
    display_df['Stock Return %'] = display_df['Stock Return %'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "N/A")
    display_df['VnIndex Return %'] = display_df['VnIndex Return %'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "N/A")
    display_df['Excess Return %'] = display_df['Excess Return %'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "N/A")
    display_df['Contribution %'] = display_df['Contribution %'].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "N/A")
    display_df['Direction'] = display_df['Direction'].apply(lambda x: f"{x:+.1f}")
    display_df['Correct'] = display_df['Correct'].apply(lambda x: "✅" if x == 1 else "❌")
    
    st.subheader("Stock-Level Contributions")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Formula explanation
    with st.expander("📐 Calculation Formula"):
        st.markdown("""
        ### How Daily Alpha is calculated:
        
        1. **Excess Return** = Stock Return - VnIndex Return
        2. **Contribution** = Direction Weight × Excess Return
        3. **Daily Alpha** = Average of all Contributions
        4. **Index Update** = Previous Index × (1 + Daily Alpha / 100)
        
        ### Direction Weights:
        - **OPF (Outperform, BUY)**: +1.0 → Bullish, positive contribution when stock beats index
        - **UPF (Underperform, SELL, AVOID)**: -1.0 → Bearish, positive contribution when stock trails index
        - **MPF (Market-Perform)**: +0.3 → Light bullish, small positive when stock beats index
        - **WATCH**: 0 → Excluded from calculation
        
        ### Correct Call:
        - **OPF**: Correct if Excess Return > 0
        - **UPF/MPF**: Correct if Excess Return < 0
        """)


if __name__ == "__main__":
    main()
