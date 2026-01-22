"""Database connection and query utilities."""
import sqlite3
import pandas as pd
from config import PRICES_DB, RECOMMENDATIONS_DB, VNINDEX_TICKER


def get_prices_connection():
    """Get connection to adjusted prices database."""
    return sqlite3.connect(PRICES_DB)


def get_recommendations_connection():
    """Get connection to recommendations database."""
    return sqlite3.connect(RECOMMENDATIONS_DB)


def get_stock_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get stock prices for a ticker within date range."""
    conn = get_prices_connection()
    query = """
        SELECT Ticker, TradeDate, AdjClose
        FROM AdjustedPrices
        WHERE Ticker = ?
          AND TradeDate BETWEEN ? AND ?
        ORDER BY TradeDate
    """
    df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
    conn.close()
    return df


def get_vnindex_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """Get VnIndex prices within date range."""
    return get_stock_prices(VNINDEX_TICKER, start_date, end_date)


def get_all_recommendations() -> pd.DataFrame:
    """Get all recommendations from database."""
    conn = get_recommendations_connection()
    query = """
        SELECT Ticker, Date, Recommendation, Analyst, Sector, Industry
        FROM RecommendationHistory
        ORDER BY Analyst, Date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    return df


def get_analyst_recommendations(analyst_email: str) -> pd.DataFrame:
    """Get all recommendations for a specific analyst."""
    conn = get_recommendations_connection()
    query = """
        SELECT Ticker, Date, Recommendation, Analyst
        FROM RecommendationHistory
        WHERE Analyst = ?
        ORDER BY Date
    """
    df = pd.read_sql_query(query, conn, params=(analyst_email,))
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    return df


def get_unique_analysts() -> list:
    """Get list of unique analysts."""
    conn = get_recommendations_connection()
    query = "SELECT DISTINCT Analyst FROM RecommendationHistory ORDER BY Analyst"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df['Analyst'].tolist()


def get_trading_dates(start_date: str, end_date: str) -> list:
    """Get list of trading dates from price database."""
    conn = get_prices_connection()
    query = """
        SELECT DISTINCT TradeDate
        FROM AdjustedPrices
        WHERE Ticker = ?
          AND TradeDate BETWEEN ? AND ?
        ORDER BY TradeDate
    """
    df = pd.read_sql_query(query, conn, params=(VNINDEX_TICKER, start_date, end_date))
    conn.close()
    return df['TradeDate'].tolist()


def get_multiple_stock_prices(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Get prices for multiple tickers, pivoted by ticker."""
    conn = get_prices_connection()
    placeholders = ','.join(['?' for _ in tickers])
    query = f"""
        SELECT Ticker, TradeDate, AdjClose
        FROM AdjustedPrices
        WHERE Ticker IN ({placeholders})
          AND TradeDate BETWEEN ? AND ?
        ORDER BY TradeDate
    """
    params = tickers + [start_date, end_date]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    # Pivot to have tickers as columns
    if not df.empty:
        df = df.pivot(index='TradeDate', columns='Ticker', values='AdjClose')
    return df
