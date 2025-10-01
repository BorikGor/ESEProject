import requests
import json
from datetime import datetime
import random

# Server endpoint (adjust if running on a different machine)
SERVER_URL = "http://192.168.1.16:5000/upload/pi2"  # or pi2

# Simulate parking slot data
slot_id = random.randint(1, 100)
slot_type = random.choice(["short", "medium", "long", "disabled", "stroller"])
status = random.choice(["occupied", "vacant"])
license_plate = None if status == "vacant" else f"TEST{random.randint(100,999)}"
timestamp = datetime.now().isoformat()

data = {
    "slot_id": slot_id,
    "slot_type": slot_type,
    "status": status,
    "license_plate": license_plate,
    "timestamp": timestamp
}

# Use any JPEG image you have
image_path = "C:/Users/Boris/Documents/AUT/!EmbeddedSoftware/Tests/pi_02.jpg"

# Send the simulated data
with open(image_path, "rb") as img_file:
    response = requests.post(
        SERVER_URL,
        files={"image": img_file},
        data={"data": json.dumps(data)}
    )

print("Sent slot", slot_id, "| Server response:", response.text)
