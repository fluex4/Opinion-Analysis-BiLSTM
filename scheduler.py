import schedule
import time
from train import full_retrain

def job():
    print("Starting weekly full retrain...")
    result = full_retrain()
    print(result)

# Schedule to run every Monday at 03:00 AM (adjust time as needed)
schedule.every().monday.at("03:00").do(job)

print("Scheduler started. Waiting for scheduled tasks...")
while True:
    schedule.run_pending()
    time.sleep(60)
