from datetime import datetime

def printr(text):
    """Prints text with a timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - {text}")
