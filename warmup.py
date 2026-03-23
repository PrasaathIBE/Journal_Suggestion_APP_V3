import requests
import os
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("EMBEDDING_API_URL", "")  # this is already the /embed endpoint
print(f"Pinging HF Space: {url}")
try:
    r = requests.post(url, json={"text": "warmup"}, timeout=180)
    print(f"HF Space awake: {r.status_code}")
except Exception as e:
    print(f"Error waking HF Space: {e}")