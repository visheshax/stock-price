import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_search_api():
    print("Testing /api/search endpoint...")
    response = client.get("/api/search?q=accenture")
    assert response.status_code == 200, f"Failed search request: {response.text}"
    data = response.json()
    assert len(data) > 0, "No search results returned for query 'accenture'"
    print(f"✅ Search API passed: Found {len(data)} results. First: {data[0]}")

def test_context_api():
    print("Testing /api/context endpoint...")
    response = client.get("/api/context?ticker=ACN")
    assert response.status_code == 200, f"Failed context request: {response.text}"
    data = response.json()
    assert "sentiment_score" in data, "No sentiment score returned"
    assert "profit_margins" in data, "No profit margins returned"
    print(f"✅ Context API passed: Sentiment={data['sentiment_score']}, Margins={data['profit_margins']}")

def test_predict_api():
    print("Testing /api/predict endpoint for ACN...")
    payload = {
        "ticker": "ACN",
        "target_date": "2026-06-30"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200, f"Failed prediction request: {response.text}"
    data = response.json()
    
    # Assertions on standard metrics
    assert data["ticker"] == "ACN"
    assert "last_price" in data
    assert "predicted_price" in data
    assert "projected_move_pct" in data
    assert len(data["chart_data"]) > 0
    
    # Verify chart data fields
    first_chart_pt = data["chart_data"][0]
    assert "date" in first_chart_pt
    assert "price" in first_chart_pt
    assert "type" in first_chart_pt
    
    print(f"✅ Predict API passed for ACN: Last={data['last_price']}, Pred={data['predicted_price']}, Move={data['projected_move_pct']}%")

if __name__ == "__main__":
    print("--- Running FastAPI Server Unit Verification ---")
    try:
        test_search_api()
        test_context_api()
        test_predict_api()
        print("\n🎉 ALL BACKEND API ENDPOINTS ARE FULLY FUNCTIONAL AND ROBUST!")
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
