import requests
import json

# Ensure this matches your running port (8002)
url = "http://localhost:8002/api/chat"

print(f"Connecting to {url}...\n")

query = """
I am planning a trip. 
1. Find the 1992 Summer Olympics host city.
2. Find the current mayor of that city.
3. Get the weather there.
"""

try:
    with requests.post(url, json={"query": query}, stream=True) as response:
        response.raise_for_status()
        
        # We iterate line by line because the server sends NDJSON
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    data = json.loads(decoded_line)
                    
                    if data.get("type") == "log":
                        print(f"🛠️  [TOOL]: {data.get('message')}")
                    elif data.get("type") == "answer":
                        print(f"\n🤖 [ANSWER]: {data.get('content')}")
                        
                except json.JSONDecodeError:
                    print(f"⚠️  RAW LINE: {decoded_line}")

except Exception as e:
    print(f"Error: {e}")