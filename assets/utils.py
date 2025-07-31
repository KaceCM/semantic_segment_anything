from datetime import datetime

def printr(text):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - {text}")
