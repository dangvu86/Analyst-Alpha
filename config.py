# Analyst Alpha Dashboard Configuration

# Direction weights for rating types
# OPF (Outperform): bullish, expect stock to BEAT VnIndex
# UPF (Underperform): bearish, expect stock to TRAIL VnIndex
# MPF (Market Perform): light bearish, upside unclear

DIRECTION_WEIGHTS = {
    # OPF ratings (+1.0) - Bullish
    'OUTPERFORM': 1.0,
    'Outperform': 1.0,
    'BUY': 1.0,
    
    # UPF ratings (-1.0) - Bearish
    'UNDER-PERFORM': -1.0,
    'UNDERPERFORM': -1.0,
    'SELL': -1.0,
    'AVOID': -1.0,
    
    # MPF ratings (+0.3) - Light bullish (positive when beat index)
    'MARKET-PERFORM': 0.3,
    
    # WATCH = 0 - Neutral, will be skipped
    'WATCH': 0,
}

# Ratings to skip (not counted in alpha calculation)
# WATCH is also skipped because weight = 0
SKIP_RATINGS = ['', 'NON-RATED', None, 'WATCH']

# Color scheme
COLORS = {
    'positive': '#27ae60',      # Green
    'negative': '#e74c3c',      # Red
    'neutral': '#95a5a6',       # Gray
    'primary': '#00e676',       # Fresh Green
    'vnindex': '#2c3e50',       # Dark gray
}

# Thresholds for classification
THRESHOLDS = {
    'high_alpha': 1.05,
    'high_hit_rate': 55,
    'good_ir': 0.3,
}

# Database paths
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRICES_DB = os.path.join(BASE_DIR, 'adjusted_prices.db')
RECOMMENDATIONS_DB = os.path.join(BASE_DIR, 'recommendation_history.db')

# VnIndex ticker name in prices database
VNINDEX_TICKER = 'VNINDEX'

# Analysis start date (YTD 2025)
from datetime import datetime
DEFAULT_START_DATE = datetime(2025, 1, 1)
DEFAULT_END_DATE = datetime.now()

# Allowed analysts (only these will be included in calculation)
ALLOWED_ANALYSTS = [
    'Thaodien@dragoncapital.com',
    'lyvu@dragoncapital.com',
    'ducle@dragoncapital.com',
    'hadao@dragoncapital.com',
    'hanguyen@dragoncapital.com',
    'quanpham@dragoncapital.com',
    'minhbui@dragoncapital.com',
    'minhtrinh@dragoncapital.com',
    'quanle@dragoncapital.com',
    'danvu@dragoncapital.com',
    'duynguyen@dragoncapital.com',
    'huyennguyen@dragoncapital.com',
    'thohoang@dragoncapital.com',
    'duongpham@dragoncapital.com',
    'hanguyenthu@dragoncapital.com',
]
