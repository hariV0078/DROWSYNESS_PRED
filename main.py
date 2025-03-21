import cv2
import requests
import numpy as np
import time
import logging
import os
import json
import base64
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SERVER_URL = "http://localhost:8000/predict"  # Will be replaced with ngrok URL
CAMERA_INDEX = 0
FRAME_INTERVAL = 0.1  # Send a frame every 100ms
JPEG_QUALITY = 70  # 0-100, higher is better quality but larger size
SAVE_RESULTS = True
OUTPUT_DIR = "drowsiness_results"

# Create output directory if needed
if SAVE_RESULTS and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logger.info(f"Created output directory: {OUTPUT_DIR}")

def update_server_url():
    """Prompt the user to enter the ngrok URL"""
    global SERVER_URL
    ngrok_url = input("Enter the ngrok URL (e.g., https://xxxx-xxx-xxx-xxx-xxx.ngrok.io/predict): ")
    if ngrok_url:
        SERVER_URL = ngrok_url
        logger.info(f"Server URL updated to: {SERVER_URL}")
    else:
        logger.warning("Using default server URL: localhost:8000")

def encode_frame(frame):
    """Encode a frame as base64 string for transmission"""
    # Compress frame to reduce size
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    # Convert to base64 string
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

def send_frame(frame):
    """Send a frame to the server for processing"""
    try:
        # Encode the frame
        encoded_frame = encode_frame(frame)
        
        # Prepare the data
        payload = {
            "image": encoded_frame,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to server
        response = requests.post(SERVER_URL, json=payload, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            logger.error(f"Error from server: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error sending frame: {str(e)}")
        return None

def run_client():
    """Main function to capture frames and send to server"""
    # Update server URL with ngrok URL
    update_server_url()
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    logger.info(f"Starting frame capture and sending to {SERVER_URL}")
    logger.info("Press Ctrl+C to stop")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture image")
                break
            
            frame_count += 1
            
            # Send frame to server
            result = send_frame(frame)
            
            if result:
                status = result.get("status", "Unknown")
                is_drowsy = result.get("is_drowsy", False)
                
                # Draw drowsiness alert on frame if needed
                if is_drowsy:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Save drowsy frame
                    if SAVE_RESULTS:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(OUTPUT_DIR, f"drowsy_{timestamp}.jpg")
                        cv2.imwrite(filename, frame)
                        logger.info(f"Drowsy event saved: {filename}")
                
                # Log status periodically
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    logger.info(f"Status: {status} | Frames: {frame_count} | FPS: {fps:.2f}")
            
            # Wait to maintain frame rate
            time.sleep(FRAME_INTERVAL)
            
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.error(f"Error during client operation: {e}")
    finally:
        # Cleanup
        cap.release()
        logger.info(f"Client stopped. Sent {frame_count} frames.")

if __name__ == "__main__":
    run_client() 
