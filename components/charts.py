"""Plotly chart components for the dashboard."""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from config import COLORS


def create_alpha_vs_vnindex_chart(analyst_data: pd.DataFrame, vnindex_data: pd.DataFrame, title: str = "Alpha Performance") -> go.Figure:
    """
    Create line chart showing analyst alpha performance in %.
    """
    fig = go.Figure()
    
    # Convert index to % performance: (index - 1) * 100
    perf_data = (analyst_data['index_value'] - 1) * 100
    
    # Build hover text manually for full control
    if 'active_tickers' in analyst_data.columns:
        hover_text = analyst_data.apply(
            lambda row: f"<b>{row['date']}</b><br><b>Performance:</b> {(row['index_value']-1)*100:+.1f}%<br><b>Active Ratings:</b><br>{row['active_tickers'] if pd.notna(row['active_tickers']) else 'N/A'}", 
            axis=1
        )
    else:
        hover_text = analyst_data.apply(
            lambda row: f"<b>{row['date']}</b><br><b>Performance:</b> {(row['index_value']-1)*100:+.1f}%", 
            axis=1
        )

    # Alpha Performance line
    fig.add_trace(go.Scatter(
        x=analyst_data['date'],
        y=perf_data,
        name='Alpha Performance',
        line=dict(color=COLORS['primary'], width=3),
        hovertext=hover_text,
        hoverinfo='text'
    ))
    
    # Reference line at 0%
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Performance (%)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        template='plotly_white',
        height=400
    )
    
    return fig


def create_daily_alpha_bars(daily_data: pd.DataFrame, days: int = 30) -> go.Figure:
    """Create bar chart for daily alpha contributions."""
    # Get last N days
    data = daily_data.tail(days)
    
    colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in data['daily_alpha']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=data['date'],
            y=data['daily_alpha'],
            marker_color=colors,
            name='Daily Alpha',
            hovertemplate='%{x}<br>Alpha: %{y:.3f}%<extra></extra>'
        )
    ])
    
    fig.add_hline(y=0, line_color="black", line_width=1)
    
    fig.update_layout(
        title=f'Daily Alpha Contribution (Last {days} Days)',
        xaxis_title='Date',
        yaxis_title='Daily Alpha (%)',
        template='plotly_white',
        height=300
    )
    
    return fig


def create_ranking_bars(scorecard_data: pd.DataFrame, metric: str = 'ytd_alpha', title: str = None) -> go.Figure:
    """Create horizontal bar chart for analyst rankings."""
    # Sort by metric
    data = scorecard_data.sort_values(metric, ascending=True)
    
    colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in data[metric]]
    
    fig = go.Figure(data=[
        go.Bar(
            x=data[metric],
            y=data['analyst_name'],
            orientation='h',
            marker_color=colors,
            text=[f"{x:+.2f}%" for x in data[metric]],
            textposition='outside',
            hovertemplate='%{y}<br>%{text}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=title or f'Rankings by {metric.replace("_", " ").title()}',
        xaxis_title=metric.replace("_", " ").title(),
        yaxis_title='',
        template='plotly_white',
        height=max(400, len(data) * 35)
    )
    
    return fig


def create_hit_rate_gauge(hit_rate: float, title: str = "Hit Rate") -> go.Figure:
    """Create a gauge chart for hit rate."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hit_rate,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': COLORS['primary']},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': COLORS['negative']},
                {'range': [50, 55], 'color': COLORS['neutral']},
                {'range': [55, 100], 'color': COLORS['positive']}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': hit_rate
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_team_overview_chart(team_data: pd.DataFrame, benchmark_label: str = None) -> go.Figure:
    """Create team alpha performance chart in %."""
    fig = go.Figure()
    
    # Add team average (converted to %)
    if 'team_avg' in team_data.columns:
        perf_data = (team_data['team_avg'] - 1) * 100
        fig.add_trace(go.Scatter(
            x=team_data['date'],
            y=perf_data,
            name='Team Average',
            line=dict(color=COLORS['primary'], width=3),
            hovertemplate='%{y:+.2f}%<extra></extra>'
        ))
    
    # Reference line at 0%
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5,
                  annotation_text=benchmark_label or "Benchmark = 0%", 
                  annotation_position="right")
    
    title = 'Team Alpha vs VNIndex'
    if benchmark_label:
        title = 'Team Alpha vs Coverage'
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Performance (%)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        template='plotly_white',
        height=450
    )
    
    return fig


def create_peer_alpha_chart(analyst_data: pd.DataFrame, title: str = "Peer Alpha Performance") -> go.Figure:
    """
    Create line chart showing analyst peer-based alpha performance in %.
    Similar to create_alpha_vs_vnindex_chart but for peer benchmark.
    """
    fig = go.Figure()
    
    # Convert index to % performance: (index - 1) * 100
    perf_data = (analyst_data['index_value'] - 1) * 100
    
    # Build hover text manually for full control
    if 'active_tickers' in analyst_data.columns:
        hover_text = analyst_data.apply(
            lambda row: f"<b>{row['date']}</b><br><b>Performance:</b> {(row['index_value']-1)*100:+.1f}%<br><b>Active Ratings:</b><br>{row['active_tickers'] if pd.notna(row['active_tickers']) else 'N/A'}", 
            axis=1
        )
    else:
        hover_text = analyst_data.apply(
            lambda row: f"<b>{row['date']}</b><br><b>Performance:</b> {(row['index_value']-1)*100:+.1f}%", 
            axis=1
        )

    # Alpha Performance line
    fig.add_trace(go.Scatter(
        x=analyst_data['date'],
        y=perf_data,
        name='Peer Alpha Performance',
        line=dict(color=COLORS['primary'], width=3),
        hovertext=hover_text,
        hoverinfo='text'
    ))
    
    # Reference line at 0% (peer average = 0)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5,
                  annotation_text="Peer Avg = 0%", annotation_position="right")
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Performance vs Peers (%)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        template='plotly_white',
        height=400
    )
    
    return fig
