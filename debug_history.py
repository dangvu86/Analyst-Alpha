
import sqlite3
import pandas as pd
import sys
import os

ALPHA_INDEX_DB = "alpha_index.db"

def load_daily_scorecard_debug(trade_date: str):
    print(f"--- Debugging for date: {trade_date} ---")
    conn = sqlite3.connect(ALPHA_INDEX_DB)
    
    # 1. Base Scorecard Data
    df = pd.read_sql_query("""
        SELECT analyst_email, index_value, cumulative_hits, cumulative_total
        FROM AnalystAlphaDaily 
        WHERE trade_date = ?
    """, conn, params=(trade_date,))
    
    print(f"Base rows: {len(df)}")
    
    if df.empty:
        print("No base data found.")
        conn.close()
        return

    # 2. Coverage & Conviction
    contrib_df = pd.read_sql_query("""
        SELECT analyst_email, direction_weight
        FROM DailyContributions
        WHERE trade_date = ?
    """, conn, params=(trade_date,))
    
    print(f"Contribution rows: {len(contrib_df)}")
    if not contrib_df.empty:
        print(contrib_df.head())
    else:
        print("No contributions found for this date!")

    metrics = []
    for email, group in contrib_df.groupby('analyst_email'):
        total = len(group)
        opf = len(group[group['direction_weight'] > 0])
        upf = len(group[group['direction_weight'] == -1.0])
        conviction = ((opf + upf) / total * 100) if total > 0 else 0
        metrics.append({
            'analyst_email': email,
            'coverage': total,
            'conviction': conviction
        })
    
    metrics_df = pd.DataFrame(metrics)
    print("Metrics DF:")
    print(metrics_df)
    
    # 3. IR
    history_df = pd.read_sql_query("""
        SELECT analyst_email, daily_alpha
        FROM AnalystAlphaDaily
        WHERE trade_date <= ?
    """, conn, params=(trade_date,))
    
    print(f"History rows: {len(history_df)}")

    ir_metrics = []
    for email, group in history_df.groupby('analyst_email'):
        if len(group) >= 20:
            avg = group['daily_alpha'].mean()
            std = group['daily_alpha'].std()
            ir = avg / std if std > 0 else 0
        else:
            ir = None
        ir_metrics.append({'analyst_email': email, 'information_ratio': ir})
    
    ir_df = pd.DataFrame(ir_metrics)
    print("IR DF:")
    print(ir_df.head())

    conn.close()

    # Merge
    if not metrics_df.empty:
        df = df.merge(metrics_df, on='analyst_email', how='left')
        print("Merged Metrics.")
    else:
        print("Skipped Metrics Merge (Empty).")
    
    if not ir_df.empty:
        df = df.merge(ir_df, on='analyst_email', how='left')
        print("Merged IR.")
        
    print("Final Columns:", df.columns)
    print(df.head())

# Run for a known date (e.g., from the screenshot user sent earlier or a recent one)
# User's screenshot showed 2026-01-12
load_daily_scorecard_debug("2026-01-12")
