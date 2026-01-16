"""KPI card components for the dashboard."""
import streamlit as st


def create_kpi_card(title: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Create a KPI metric card."""
    st.metric(
        label=title,
        value=value,
        delta=delta,
        delta_color=delta_color
    )


def create_kpi_row(kpis: list):
    """
    Create a row of KPI cards.
    
    kpis: list of dicts with keys: title, value, delta (optional), delta_color (optional)
    """
    cols = st.columns(len(kpis))
    
    for i, kpi in enumerate(kpis):
        with cols[i]:
            create_kpi_card(
                title=kpi.get('title', ''),
                value=kpi.get('value', ''),
                delta=kpi.get('delta'),
                delta_color=kpi.get('delta_color', 'normal')
            )


def format_alpha_index(value: float) -> str:
    """Format alpha index value for display."""
    return f"{value:.2f}"


def format_percentage(value: float) -> str:
    """Format percentage value for display."""
    return f"{value:+.2f}%"


def format_hit_rate(value: float) -> str:
    """Format hit rate for display."""
    return f"{value:.1f}%"


def format_ir(value: float) -> str:
    """Format information ratio for display."""
    if value is None:
        return "N/A"
    return f"{value:.2f}"
