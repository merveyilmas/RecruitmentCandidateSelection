import requests
import json
import random

BASE_URL = "http://localhost:8000"

def test_api():
    print("Testing Candidate Selection API...")
    
    # Test root endpoint
    print("\n1. Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test model info endpoint
    print("\n2. Testing model info endpoint...")
    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test prediction endpoint with multiple test cases
    print("\n3. Testing prediction endpoint...")
    test_cases = [
        {"experience_years": 2.5, "technical_score": 75},
        {"experience_years": 5.0, "technical_score": 85},
        {"experience_years": 1.0, "technical_score": 60},
        {"experience_years": 8.0, "technical_score": 90}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {test_case}")
        response = requests.post(f"{BASE_URL}/predict", json=test_case)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

if __name__ == "__main__":
    test_api() 