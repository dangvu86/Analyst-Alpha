"""
Pre-calculation Script for Analyst Alpha Index
===============================================

Chạy script này để tính sẵn Alpha Index cho tất cả analysts.
Kết quả lưu vào alpha_index.db để app đọc nhanh.

Usage:
    python precalculate.py

Chạy mỗi ngày hoặc khi có data mới.
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DIRECTION_WEIGHTS, SKIP_RATINGS, VNINDEX_TICKER,
    PRICES_DB, RECOMMENDATIONS_DB, ALLOWED_ANALYSTS
)
from data.database import (
    get_all_recommendations, get_unique_analysts,
    get_trading_dates, get_multiple_stock_prices
)
from data.calculations import (
    calculate_daily_returns, calculate_analyst_alpha_index,
    calculate_analyst_summary, get_active_ratings_on_date,
    calculate_analyst_daily_alpha
)

# Output database
ALPHA_INDEX_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alpha_index.db')


def create_alpha_index_db():
    """Create the alpha index database with required tables."""
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    cursor = conn.cursor()
    
    # Table for analyst daily alpha history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS AnalystAlphaDaily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analyst_email TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            daily_alpha REAL,
            index_value REAL,
            hits INTEGER,
            total INTEGER,
            cumulative_hits INTEGER,
            cumulative_total INTEGER,
            UNIQUE(analyst_email, trade_date)
        )
    """)
    
    # Table for analyst summary (scorecard)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS AnalystSummary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analyst_email TEXT NOT NULL,
            analyst_name TEXT,
            as_of_date TEXT NOT NULL,
            index_value REAL,
            ytd_alpha REAL,
            hit_rate REAL,
            information_ratio REAL,
            conviction REAL,
            coverage INTEGER,
            opf_count INTEGER,
            upf_count INTEGER,
            mpf_count INTEGER,
            UNIQUE(analyst_email, as_of_date)
        )
    """)
    
    # Table for meta info
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS CalculationMeta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            last_updated TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            total_analysts INTEGER,
            total_records INTEGER
        )
    """)
    
    # Create indexes for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alpha_analyst ON AnalystAlphaDaily(analyst_email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alpha_date ON AnalystAlphaDaily(trade_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_summary_analyst ON AnalystSummary(analyst_email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_summary_date ON AnalystSummary(as_of_date)")
    
    # NEW: Table for daily stock-level contributions (for verification)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS DailyContributions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analyst_email TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            recommendation TEXT,
            direction_weight REAL,
            stock_return REAL,
            vnindex_return REAL,
            excess_return REAL,
            contribution REAL,
            is_correct INTEGER,
            UNIQUE(analyst_email, trade_date, ticker)
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_contrib_analyst ON DailyContributions(analyst_email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_contrib_date ON DailyContributions(trade_date)")
    
    conn.commit()
    conn.close()
    print("✅ Created alpha_index.db with tables")


def load_source_data(start_date_str: str, end_date_str: str) -> dict:
    """Load all required source data."""
    print(f"📂 Loading source data from {start_date_str} to {end_date_str}...")
    
    # Load recommendations
    recommendations = get_all_recommendations()
    print(f"   - {len(recommendations)} recommendations loaded")
    
    # Get unique tickers
    tickers = recommendations['Ticker'].unique().tolist()
    
    # Get trading dates
    trading_dates = get_trading_dates(start_date_str, end_date_str)
    print(f"   - {len(trading_dates)} trading dates")
    
    # Load prices for all tickers + VNINDEX
    all_tickers = list(set(tickers + [VNINDEX_TICKER]))
    prices_df = get_multiple_stock_prices(all_tickers, start_date_str, end_date_str)
    print(f"   - {len(prices_df)} price records loaded")
    
    # Calculate returns
    if not prices_df.empty:
        returns_df = calculate_daily_returns(prices_df)
        vnindex_returns = returns_df[VNINDEX_TICKER] if VNINDEX_TICKER in returns_df.columns else pd.Series()
    else:
        returns_df = pd.DataFrame()
        vnindex_returns = pd.Series()
    
    return {
        'recommendations': recommendations,
        'prices': prices_df,
        'returns': returns_df,
        'vnindex_returns': vnindex_returns,
        'trading_dates': trading_dates,
        'analysts': ALLOWED_ANALYSTS  # Use filtered list instead of all analysts
    }


def calculate_and_save(start_date_str: str, end_date_str: str):
    """Calculate Alpha Index for all analysts and save to database."""
    
    # Load source data
    data = load_source_data(start_date_str, end_date_str)
    
    if data['returns'].empty:
        print("❌ No price data available!")
        return
    
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    # Clear existing data for this date range
    conn.execute("DELETE FROM AnalystAlphaDaily WHERE trade_date BETWEEN ? AND ?", 
                 (start_date_str, end_date_str))
    conn.execute("DELETE FROM AnalystSummary WHERE as_of_date = ?", (end_date_str,))
    conn.execute("DELETE FROM DailyContributions WHERE trade_date BETWEEN ? AND ?",
                 (start_date_str, end_date_str))
    conn.commit()
    
    total_records = 0
    total_contributions = 0
    valid_analysts = 0
    
    print(f"\n🔄 Calculating Alpha Index for {len(data['analysts'])} analysts...")
    
    for i, analyst in enumerate(data['analysts']):
        if not analyst or pd.isna(analyst):
            continue
        
        print(f"   [{i+1}/{len(data['analysts'])}] Processing {analyst}...", end=" ")
        
        # Calculate alpha history
        alpha_history = calculate_analyst_alpha_index(
            analyst, start_date_str, end_date_str,
            data['recommendations'], data['returns'],
            data['vnindex_returns'], data['trading_dates']
        )
        
        if alpha_history.empty:
            print("⏭️ No data")
            continue
        
        # Save daily alpha to database
        for _, row in alpha_history.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO AnalystAlphaDaily 
                (analyst_email, trade_date, daily_alpha, index_value, hits, total, cumulative_hits, cumulative_total)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analyst,
                row['date'],
                row['daily_alpha'],
                row['index_value'],
                row['hits'],
                row['total'],
                row['cumulative_hits'],
                row['cumulative_total']
            ))
        
        # NEW: Save stock-level contributions for each trading day
        for trade_date in data['trading_dates']:
            if trade_date < start_date_str or trade_date > end_date_str:
                continue
            
            # Calculate daily details with stock-level breakdown
            daily_result = calculate_analyst_daily_alpha(
                analyst, trade_date, data['recommendations'], 
                data['returns'], data['vnindex_returns']
            )
            
            # Save each stock contribution
            for contrib in daily_result['contributions']:
                conn.execute("""
                    INSERT OR REPLACE INTO DailyContributions
                    (analyst_email, trade_date, ticker, recommendation, direction_weight,
                     stock_return, vnindex_return, excess_return, contribution, is_correct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analyst,
                    trade_date,
                    contrib['ticker'],
                    contrib['recommendation'],
                    contrib['direction'],
                    contrib['stock_return'],
                    contrib['vnindex_return'],
                    contrib['excess_return'],
                    contrib['contribution'],
                    1 if contrib['is_correct'] else 0
                ))
                total_contributions += 1
        
        
        total_records += len(alpha_history)
        
        # Calculate summary
        summary = calculate_analyst_summary(analyst, alpha_history)
        
        # Get rating distribution
        active_ratings = get_active_ratings_on_date(
            data['recommendations'], end_date_str, analyst
        )
        
        opf_count = len(active_ratings[active_ratings['Direction'] > 0]) if not active_ratings.empty else 0
        upf_count = len(active_ratings[active_ratings['Direction'] == -1.0]) if not active_ratings.empty else 0
        mpf_count = len(active_ratings[active_ratings['Direction'] == -0.3]) if not active_ratings.empty else 0
        total_count = len(active_ratings) if not active_ratings.empty else 0
        conviction = ((opf_count + upf_count) / total_count * 100) if total_count > 0 else 0
        
        analyst_name = analyst.split('@')[0] if '@' in analyst else analyst
        
        # Save summary
        conn.execute("""
            INSERT OR REPLACE INTO AnalystSummary
            (analyst_email, analyst_name, as_of_date, index_value, ytd_alpha, hit_rate, 
             information_ratio, conviction, coverage, opf_count, upf_count, mpf_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analyst,
            analyst_name,
            end_date_str,
            summary['index_value'],
            summary['ytd_alpha'],
            summary['hit_rate'],
            summary['information_ratio'],
            conviction,
            total_count,
            opf_count,
            upf_count,
            mpf_count
        ))
        
        valid_analysts += 1
        print(f"✅ {len(alpha_history)} days, Index={summary['index_value']:.2f}")
    
    # Save meta info
    conn.execute("""
        INSERT INTO CalculationMeta (last_updated, start_date, end_date, total_analysts, total_records)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        start_date_str,
        end_date_str,
        valid_analysts,
        total_records
    ))
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ DONE! Saved {total_records} records for {valid_analysts} analysts to alpha_index.db")


def main():
    """Main entry point."""
    print("=" * 60)
    print("ANALYST ALPHA INDEX PRE-CALCULATION")
    print("=" * 60)
    
    # Create database
    create_alpha_index_db()
    
    # Default: From 2020-01-01 to today
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    print(f"\n📅 Period: {start_date_str} to {end_date_str}")
    
    # Calculate and save
    calculate_and_save(start_date_str, end_date_str)
    
    print(f"\n📁 Output: {ALPHA_INDEX_DB}")
    print("🚀 You can now push alpha_index.db to GitHub!")


if __name__ == "__main__":
    main()
