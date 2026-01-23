
import pandas as pd
import sys
import os

# Mock streamlit before importing app
import types
from unittest.mock import MagicMock
sys.modules['streamlit'] = MagicMock()

# Import the function from app.py
from app import load_daily_scorecard, load_peer_daily_scorecard

def verify_app_logic():
    print("--- Verifying app.py load_daily_scorecard Logic ---")
    
    # Test date known to have data from debug_history.py
    test_date = "2026-01-12" 
    
    print(f"Testing date: {test_date}")
    
    try:
        df = load_daily_scorecard(test_date)
        print(f"Result rows: {len(df)}")
        if not df.empty:
            cols = ['analyst_email', 'coverage', 'conviction', 'information_ratio']
            print(df[cols].head())
            print("\nCheck for N/A values:")
            print(df[cols].isna().sum())
            
            # Check for specific row
            row = df[df['analyst_email'] == 'Thaodien@dragoncapital.com']
            if not row.empty:
                 print("\nSample Row (Thaodien):")
                 print(row[cols])
            else:
                 print("\nThaodien row not found")
        else:
            print("DataFrame is empty")
            
    except Exception as e:
        print(f"Error calling load_daily_scorecard: {e}")

if __name__ == "__main__":
    verify_app_logic()
