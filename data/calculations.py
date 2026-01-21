"""Alpha Index calculation functions."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import DIRECTION_WEIGHTS, SKIP_RATINGS, VNINDEX_TICKER


def get_direction_weight(recommendation: str) -> float:
    """Get direction weight for a recommendation. Returns None if should skip."""
    if recommendation in SKIP_RATINGS or pd.isna(recommendation):
        return None
    return DIRECTION_WEIGHTS.get(recommendation, None)


def calculate_daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from price dataframe."""
    returns = prices_df.pct_change() * 100  # Convert to percentage
    return returns


def get_active_ratings_on_date(recommendations_df: pd.DataFrame, as_of_date, analyst_email: str = None) -> pd.DataFrame:
    """
    Get active ratings as of a specific date.
    Active rating = the most recent recommendation for each ticker before or on that date.
    
    IMPORTANT: If a ticker has been covered by a DIFFERENT analyst after this analyst's rating,
    that ticker is NO LONGER active for this analyst, even if the original rating is still "valid".
    
    Example:
    - Analyst A rates VIC on 2022-10-01
    - Analyst B rates VIC on 2024-01-01
    - On date 2024-06-01: VIC is active for B, NOT for A
    """
    df = recommendations_df.copy()
    
    # Filter to only recommendations on or before the date
    df = df[df['Date'] <= as_of_date]
    
    if df.empty:
        return pd.DataFrame()
    
    # First, find the CURRENT owner of each ticker (most recent recommendation from ANY analyst)
    df_sorted = df.sort_values('Date', ascending=False)
    current_owners = df_sorted.groupby('Ticker').first().reset_index()[['Ticker', 'Analyst', 'Date']]
    current_owners = current_owners.rename(columns={'Analyst': 'CurrentOwner', 'Date': 'OwnerDate'})
    
    # If filtering by analyst, get only tickers where this analyst is the CURRENT owner
    if analyst_email:
        # Get tickers currently owned by this analyst
        owned_tickers = current_owners[current_owners['CurrentOwner'] == analyst_email]['Ticker'].tolist()
        
        if not owned_tickers:
            return pd.DataFrame()
        
        # Get this analyst's most recent recommendation for each owned ticker
        analyst_df = df[df['Analyst'] == analyst_email]
        analyst_df = analyst_df[analyst_df['Ticker'].isin(owned_tickers)]
        
        if analyst_df.empty:
            return pd.DataFrame()
        
        analyst_df = analyst_df.sort_values('Date', ascending=False)
        active_ratings = analyst_df.groupby('Ticker').first().reset_index()
    else:
        # No analyst filter - get the current owner's rating for each ticker
        active_ratings = df_sorted.groupby('Ticker').first().reset_index()
    
    # Filter out skipped ratings
    active_ratings['Direction'] = active_ratings['Recommendation'].apply(get_direction_weight)
    active_ratings = active_ratings[active_ratings['Direction'].notna()]
    
    return active_ratings


def calculate_analyst_daily_alpha(
    analyst_email: str,
    trade_date: str,
    recommendations_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    vnindex_returns: pd.Series
) -> dict:
    """
    Calculate daily alpha for an analyst on a specific date.
    
    Returns dict with:
        - daily_alpha: the weighted average alpha contribution
        - hits: number of correct calls
        - total: total number of rated stocks
        - contributions: list of individual contributions
    """
    # Get active ratings for this analyst on this date
    active_ratings = get_active_ratings_on_date(recommendations_df, trade_date, analyst_email)
    
    if active_ratings.empty:
        return {'daily_alpha': 0, 'hits': 0, 'total': 0, 'contributions': []}
    
    # Get VnIndex return for this date
    if trade_date not in vnindex_returns.index:
        return {'daily_alpha': 0, 'hits': 0, 'total': 0, 'contributions': []}
    
    vnindex_return = vnindex_returns.loc[trade_date]
    
    # Calculate contribution for each rated stock
    contributions = []
    hits = 0
    
    for _, rating in active_ratings.iterrows():
        ticker = rating['Ticker']
        direction = rating['Direction']
        recommendation = rating['Recommendation']
        
        # Get stock return for this date
        if ticker not in prices_df.columns or trade_date not in prices_df.index:
            continue
        
        stock_return = prices_df.loc[trade_date, ticker]
        
        if pd.isna(stock_return) or pd.isna(vnindex_return):
            continue
        
        # Calculate excess return and contribution
        excess_return = stock_return - vnindex_return
        contribution = direction * excess_return
        
        # Check if call was correct
        is_correct = False
        if direction > 0 and excess_return > 0:  # OPF and beat index
            is_correct = True
        elif direction < 0 and excess_return < 0:  # UPF/MPF and trail index
            is_correct = True
        
        if is_correct:
            hits += 1
        
        contributions.append({
            'ticker': ticker,
            'recommendation': recommendation,
            'direction': direction,
            'stock_return': stock_return,
            'vnindex_return': vnindex_return,
            'excess_return': excess_return,
            'contribution': contribution,
            'is_correct': is_correct
        })
    
    # Calculate daily alpha as average contribution
    if contributions:
        daily_alpha = np.mean([c['contribution'] for c in contributions])
        total = len(contributions)
    else:
        daily_alpha = 0
        total = 0
    
    return {
        'daily_alpha': daily_alpha,
        'hits': hits,
        'total': total,
        'contributions': contributions
    }


def calculate_analyst_daily_alpha_peer(
    analyst_email: str,
    trade_date: str,
    recommendations_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    all_active_tickers_returns: dict = None
) -> dict:
    """
    Calculate daily alpha for an analyst using PEER BENCHMARK.
    
    Benchmark = Average return of all active stocks in the analyst's coverage list.
    
    Args:
        analyst_email: Analyst email
        trade_date: Trading date
        recommendations_df: All recommendations
        prices_df: Price returns dataframe (already calculated returns)
        all_active_tickers_returns: Optional pre-calculated dict of {ticker: return} for all active tickers
    
    Returns dict with:
        - daily_alpha: the weighted average alpha contribution
        - hits: number of correct calls
        - total: total number of rated stocks
        - contributions: list of individual contributions
        - avg_peer_return: the average return of all active stocks (for reference)
    """
    # Get active ratings for this analyst on this date
    active_ratings = get_active_ratings_on_date(recommendations_df, trade_date, analyst_email)
    
    if active_ratings.empty:
        return {'daily_alpha': 0, 'hits': 0, 'total': 0, 'contributions': [], 'avg_peer_return': 0}
    
    # Collect returns for all active tickers of this analyst
    ticker_returns = {}
    for _, rating in active_ratings.iterrows():
        ticker = rating['Ticker']
        if ticker in prices_df.columns and trade_date in prices_df.index:
            ret = prices_df.loc[trade_date, ticker]
            if pd.notna(ret):
                ticker_returns[ticker] = ret
    
    if not ticker_returns:
        return {'daily_alpha': 0, 'hits': 0, 'total': 0, 'contributions': [], 'avg_peer_return': 0}
    
    # Calculate average peer return (equal-weighted)
    avg_peer_return = np.mean(list(ticker_returns.values()))
    
    # Calculate contribution for each rated stock
    contributions = []
    hits = 0
    
    for _, rating in active_ratings.iterrows():
        ticker = rating['Ticker']
        direction = rating['Direction']
        recommendation = rating['Recommendation']
        
        # Get stock return for this date
        if ticker not in ticker_returns:
            continue
        
        stock_return = ticker_returns[ticker]
        
        # Calculate excess return vs peer average
        excess_return = stock_return - avg_peer_return
        contribution = direction * excess_return
        
        # Check if call was correct
        is_correct = False
        if direction > 0 and excess_return > 0:  # OPF and beat peers
            is_correct = True
        elif direction < 0 and excess_return < 0:  # UPF/MPF and trail peers
            is_correct = True
        
        if is_correct:
            hits += 1
        
        contributions.append({
            'ticker': ticker,
            'recommendation': recommendation,
            'direction': direction,
            'stock_return': stock_return,
            'peer_return': avg_peer_return,
            'excess_return': excess_return,
            'contribution': contribution,
            'is_correct': is_correct
        })
    
    # Calculate daily alpha as average contribution
    if contributions:
        daily_alpha = np.mean([c['contribution'] for c in contributions])
        total = len(contributions)
    else:
        daily_alpha = 0
        total = 0
    
    return {
        'daily_alpha': daily_alpha,
        'hits': hits,
        'total': total,
        'contributions': contributions,
        'avg_peer_return': avg_peer_return
    }


def calculate_analyst_alpha_index(
    analyst_email: str,
    start_date: str,
    end_date: str,
    recommendations_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    vnindex_returns: pd.Series,
    trading_dates: list
) -> pd.DataFrame:
    """
    Calculate Alpha Index history for an analyst over a date range.
    
    Returns DataFrame with columns:
        - date: trading date
        - daily_alpha: daily alpha contribution
        - index_value: cumulative index value (starts at 100)
        - hits: daily hit count
        - total: daily total rated stocks
        - cumulative_hits: running total of hits
        - cumulative_total: running total of rated stocks
    """
    results = []
    index_value = 1.0
    cumulative_hits = 0
    cumulative_total = 0
    
    for date in trading_dates:
        if date < start_date or date > end_date:
            continue
        
        # Calculate daily metrics
        daily_result = calculate_analyst_daily_alpha(
            analyst_email, date, recommendations_df, prices_df, vnindex_returns
        )
        
        daily_alpha = daily_result['daily_alpha']
        hits = daily_result['hits']
        total = daily_result['total']
        
        # Update index value (compound)
        index_value = index_value * (1 + daily_alpha / 100)
        
        # Update cumulative counts
        cumulative_hits += hits
        cumulative_total += total
        
        results.append({
            'date': date,
            'daily_alpha': daily_alpha,
            'index_value': index_value,
            'hits': hits,
            'total': total,
            'cumulative_hits': cumulative_hits,
            'cumulative_total': cumulative_total
        })
    
    return pd.DataFrame(results)


def calculate_analyst_alpha_index_peer(
    analyst_email: str,
    start_date: str,
    end_date: str,
    recommendations_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    trading_dates: list
) -> pd.DataFrame:
    """
    Calculate Alpha Index history for an analyst using PEER BENCHMARK.
    
    Returns DataFrame with columns:
        - date: trading date
        - daily_alpha: daily alpha contribution
        - index_value: cumulative index value (starts at 1.0)
        - hits: daily hit count
        - total: daily total rated stocks
        - cumulative_hits: running total of hits
        - cumulative_total: running total of rated stocks
    """
    results = []
    index_value = 1.0
    cumulative_hits = 0
    cumulative_total = 0
    
    for date in trading_dates:
        if date < start_date or date > end_date:
            continue
        
        # Calculate daily metrics using peer benchmark
        daily_result = calculate_analyst_daily_alpha_peer(
            analyst_email, date, recommendations_df, prices_df
        )
        
        daily_alpha = daily_result['daily_alpha']
        hits = daily_result['hits']
        total = daily_result['total']
        
        # Update index value (compound)
        index_value = index_value * (1 + daily_alpha / 100)
        
        # Update cumulative counts
        cumulative_hits += hits
        cumulative_total += total
        
        results.append({
            'date': date,
            'daily_alpha': daily_alpha,
            'index_value': index_value,
            'hits': hits,
            'total': total,
            'cumulative_hits': cumulative_hits,
            'cumulative_total': cumulative_total
        })
    
    return pd.DataFrame(results)


def calculate_team_alpha_index(
    start_date: str,
    end_date: str,
    recommendations_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    vnindex_returns: pd.Series,
    trading_dates: list
) -> pd.DataFrame:
    """
    Calculate Team average Alpha Index (average of all analysts).
    """
    # Get unique analysts
    analysts = recommendations_df['Analyst'].unique()
    
    # Calculate for each analyst
    analyst_indices = {}
    for analyst in analysts:
        df = calculate_analyst_alpha_index(
            analyst, start_date, end_date,
            recommendations_df, prices_df, vnindex_returns, trading_dates
        )
        if not df.empty:
            analyst_indices[analyst] = df.set_index('date')['index_value']
    
    if not analyst_indices:
        return pd.DataFrame()
    
    # Combine into single DataFrame and calculate mean
    combined = pd.DataFrame(analyst_indices)
    combined['team_avg'] = combined.mean(axis=1)
    combined = combined.reset_index()
    combined = combined.rename(columns={'index': 'date'})
    
    return combined


def calculate_analyst_summary(
    analyst_email: str,
    alpha_history: pd.DataFrame
) -> dict:
    """
    Calculate summary metrics for an analyst from their alpha history.
    """
    if alpha_history.empty:
        return {
            'analyst_email': analyst_email,
            'index_value': 1.0,
            'ytd_alpha': 0,
            'hit_rate': 0,
            'information_ratio': None,
            'daily_alphas': []
        }
    
    # Latest values
    latest = alpha_history.iloc[-1]
    index_value = latest['index_value']
    ytd_alpha = (index_value - 1) * 100
    
    # Hit rate
    cumulative_hits = latest['cumulative_hits']
    cumulative_total = latest['cumulative_total']
    hit_rate = (cumulative_hits / cumulative_total * 100) if cumulative_total > 0 else 0
    
    # Information Ratio (need at least 20 days of data)
    daily_alphas = alpha_history['daily_alpha'].tolist()
    if len(daily_alphas) >= 20:
        mean_alpha = np.mean(daily_alphas)
        std_alpha = np.std(daily_alphas)
        information_ratio = mean_alpha / std_alpha if std_alpha > 0 else 0
    else:
        information_ratio = None
    
    return {
        'analyst_email': analyst_email,
        'index_value': index_value,
        'ytd_alpha': ytd_alpha,
        'hit_rate': hit_rate,
        'information_ratio': information_ratio,
        'daily_alphas': daily_alphas
    }
