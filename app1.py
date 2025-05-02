# import numpy as np
# import cv2

# fire_cascade = cv2.CascadeClassifier('fire_detection.xml')

# cap = cv2.VideoCapture(0)

# while(True):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     fire = fire_cascade.detectMultiScale(frame, 1.2, 5)

#     for (x,y,w,h) in fire:
#         cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break













import numpy as np
import cv2
import time
import threading
import os
import subprocess
from datetime import datetime
import sys
import platform

class FireDetectionSystem:
    def __init__(self, cascade_path='fire_detection.xml', camera_index=0, 
                 confidence_threshold=0.7, save_detections=True):
        # Core parameters
        self.cascade_path = cascade_path
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.save_detections = save_detections
        
        # System state
        self.fire_detected = False
        self.detection_count = 0
        self.false_positive_count = 0
        self.alarm_active = False
        self.start_time = time.time()
        self.last_detection_time = 0
        self.consecutive_detections = 0
        self.alarm_silenced = False
        
        # Create built-in alarm sound
        self.create_alarm_sound()
        
        # Output directory for saving detected fire frames
        self.output_dir = 'fire_detections'
        if self.save_detections and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Load cascade classifier
        self.load_cascade()
        
    def create_alarm_sound(self):
        """Create a simple alarm beep sound file using system commands"""
        self.alarm_dir = "alarm_sounds"
        if not os.path.exists(self.alarm_dir):
            os.makedirs(self.alarm_dir)
            
        # Create a simple alarm sound using the 'say' command on macOS
        self.alarm_file = os.path.join(self.alarm_dir, "alarm_alert.aiff")
        self.voice_file = os.path.join(self.alarm_dir, "voice_alert.aiff")
        
        # Create voice alert
        voice_cmd = ['say', '-v', 'Daniel', '-o', self.voice_file, 'Fire detected! Fire detected! Please evacuate immediately!']
        try:
            subprocess.run(voice_cmd, check=True)
            print(f"Created voice alert file: {self.voice_file}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not create voice alert: {e}")
            self.voice_file = None
            
        # Create beep alert using afplay if on macOS 
        if platform.system() == 'Darwin':  # macOS
            # We'll use the system alert sound
            self.alarm_file = "/System/Library/Sounds/Sosumi.aiff"
            if not os.path.exists(self.alarm_file):
                self.alarm_file = "/System/Library/Sounds/Ping.aiff"  # Fallback
        else:
            # For other systems, we won't have a reliable alarm sound
            self.alarm_file = None
            print("Warning: Reliable alarm sound may not be available on this platform")
            
        print(f"Alarm system initialized")
        
    def load_cascade(self):
        """Load the Haar cascade classifier for fire detection"""
        try:
            self.fire_cascade = cv2.CascadeClassifier(self.cascade_path)
            if self.fire_cascade.empty():
                raise Exception(f"Failed to load cascade classifier from {self.cascade_path}")
            print(f"Cascade classifier loaded successfully from {self.cascade_path}")
        except Exception as e:
            print(f"Error loading cascade classifier: {e}")
            sys.exit(1)
    
    def calculate_confidence(self, detection, roi_color):
        """Calculate confidence score based on multiple factors"""
        x, y, w, h = detection
        
        # Factor 1: Size - larger detections are more likely to be real fires
        size_factor = min(1.0, (w * h) / 10000)
        
        # Factor 2: Color analysis - check for fire-like colors
        color_score = self.analyze_fire_colors(roi_color)
        
        # Factor 3: Brightness - fires are typically bright
        brightness_score = self.analyze_brightness(roi_color)
        
        # Combine factors
        confidence = (size_factor * 0.4) + (color_score * 0.4) + (brightness_score * 0.2)
        
        return confidence
    
    def analyze_fire_colors(self, roi):
        """Analyze if the region contains fire-like colors"""
        # Convert to HSV for better color analysis
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define fire color ranges in HSV
        # Lower range (reddish-orange)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        # Upper range (reddish)
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        # Yellow range
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        # Create masks and combine
        mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        mask3 = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
        
        # Calculate percentage of fire-like pixels
        fire_pixel_ratio = np.count_nonzero(mask) / (roi.shape[0] * roi.shape[1])
        
        return min(1.0, fire_pixel_ratio * 2)  # Scale up but cap at 1.0
    
    def analyze_brightness(self, roi):
        """Analyze brightness of the region"""
        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness
        avg_brightness = np.mean(gray_roi) / 255.0
        
        return avg_brightness
    
    def play_alarm(self):
        """Play alarm sound"""
        if not self.alarm_active and not self.alarm_silenced:
            self.alarm_active = True
            threading.Thread(target=self._play_alarm_thread).start()
    
    def _play_alarm_thread(self):
        """Thread to play the alarm sound with multiple fallbacks"""
        try:
            print("⚠️ ALARM ACTIVATED! ⚠️")
            
            # First try to use the voice alert
            if self.voice_file and os.path.exists(self.voice_file):
                print("Playing voice alert...")
                result = subprocess.run(['afplay', self.voice_file], check=False)
                if result.returncode != 0:
                    print("Warning: Voice alert failed to play")
            
            # Then try the alarm sound
            if self.alarm_file and os.path.exists(self.alarm_file):
                print("Playing alarm sound...")
                # Try to increase volume
                try:
                    subprocess.run(['osascript', '-e', 'set volume output volume 80'], check=False)
                except:
                    pass
                
                # Play alarm sound multiple times
                for _ in range(3):
                    if self.alarm_silenced or not self.fire_detected:
                        break
                    subprocess.run(['afplay', self.alarm_file], check=False)
                    time.sleep(0.5)
            else:
                # Last resort: Use the terminal bell
                print("\a\a\a")  # Terminal bell
                print("WARNING: Using terminal bell as fallback alarm")
                time.sleep(0.5)
                print("\a\a\a")
            
            # Visual confirmation in terminal
            print("🔔 Alarm activated! Fire detected! 🔥")
            
            # Check if sound likely worked
            if platform.system() == 'Darwin':
                output_volume = subprocess.check_output(['osascript', '-e', 'output volume of (get volume settings)'])
                if int(output_volume) == 0:
                    print("⚠️ WARNING: System volume appears to be muted! Check your volume settings! ⚠️")
                
            self.alarm_active = False
            
        except Exception as e:
            print(f"⚠️ Error playing alarm: {e}")
            # Last resort emergency alert
            print("\n\n")
            print("*" * 50)
            print("⚠️ EMERGENCY: FIRE DETECTED! ⚠️")
            print("*" * 50)
            print("\n\n")
            self.alarm_active = False
    
    def toggle_alarm_silence(self):
        """Toggle alarm silence mode"""
        self.alarm_silenced = not self.alarm_silenced
        return self.alarm_silenced
    
    def send_notification(self, image_path=None):
        """Send macOS notification"""
        try:
            title = "🔥 FIRE DETECTED! 🔥"
            message = f"Fire detected at {datetime.now().strftime('%H:%M:%S')} with {self.consecutive_detections} consecutive detections"
            
            # Use macOS notification center
            cmd = f'display notification "{message}" with title "{title}" sound name "Sosumi"'
            subprocess.run(["osascript", "-e", cmd], check=False)
            
            print(f"Notification sent: {message}")
        except Exception as e:
            print(f"Error sending notification: {e}")
    
    def save_detection_image(self, frame):
        """Save the frame with fire detection"""
        if self.save_detections:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"fire_detection_{timestamp}.jpg")
            cv2.imwrite(filepath, frame)
            return filepath
        return None
    
    def run(self):
        """Main function to run the fire detection system"""
        cap = cv2.VideoCapture(self.camera_index)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        print("\n=== Fire Detection System Started ===")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to silence/unsilence alarm")
        print("- Press 'a' to test alarm")
        print("=====================================\n")
        
        # For FPS calculation
        prev_frame_time = 0
        new_frame_time = 0
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame. Exiting...")
                break
            
            # Calculate FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 30
            prev_frame_time = new_frame_time
            
            # Create a copy for displaying
            display_frame = frame.copy()
            
            # Convert to grayscale for cascade detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect fire using cascade classifier
            fire_detections = self.fire_cascade.detectMultiScale(
                frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Process detections
            current_fire_detected = False
            fire_confidence = 0.0
            
            for detection in fire_detections:
                x, y, w, h = detection
                
                # Extract region of interest
                roi_color = frame[y:y+h, x:x+w]
                
                # Skip if ROI is empty
                if roi_color.size == 0:
                    continue
                
                # Calculate confidence
                confidence = self.calculate_confidence(detection, roi_color)
                fire_confidence = max(fire_confidence, confidence)
                
                # If confidence is high enough
                if confidence >= self.confidence_threshold:
                    current_fire_detected = True
                    self.detection_count += 1
                    
                    # Draw rectangle with confidence-based color
                    # Green to red based on confidence
                    green = int(255 * (1 - confidence))
                    red = int(255 * confidence)
                    color = (0, green, red)
                    thickness = 2
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, thickness)
                    
                    # Add text with confidence percentage
                    confidence_text = f"{confidence:.2f}"
                    cv2.putText(display_frame, confidence_text, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Update fire detection state
            if current_fire_detected:
                self.fire_detected = True
                self.last_detection_time = time.time()
                self.consecutive_detections += 1
                
                # If we have multiple consecutive detections, consider it a confirmed fire
                if self.consecutive_detections >= 3:
                    # Save the image
                    image_path = self.save_detection_image(display_frame)
                    
                    # Play alarm and send notification (not too frequently)
                    if not self.alarm_active and time.time() - self.last_detection_time > 3:
                        self.play_alarm()
                        self.send_notification(image_path)
                    
                    # Add a warning label - pulsating effect
                    pulse = (np.sin(time.time() * 10) + 1) * 0.5  # Pulsating value between 0 and 1
                    text_color = (0, int(255 * (1-pulse)), int(255 * pulse))
                    alarm_status = " [SILENCED]" if self.alarm_silenced else ""
                    
                    cv2.putText(display_frame, f"FIRE DETECTED!{alarm_status}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                    
                    # Draw fire confidence meter
                    meter_length = 200
                    meter_height = 20
                    meter_x = 10
                    meter_y = 50
                    
                    # Background (gray)
                    cv2.rectangle(display_frame, (meter_x, meter_y),
                                (meter_x + meter_length, meter_y + meter_height),
                                (100, 100, 100), -1)
                    
                    # Foreground (colored based on confidence)
                    filled_length = int(meter_length * fire_confidence)
                    green_val = int(255 * (1 - fire_confidence))
                    red_val = int(255 * fire_confidence)
                    cv2.rectangle(display_frame, (meter_x, meter_y),
                                (meter_x + filled_length, meter_y + meter_height),
                                (0, green_val, red_val), -1)
                    
                    # Text
                    cv2.putText(display_frame, f"Fire Confidence: {fire_confidence:.2f}",
                                (meter_x, meter_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                self.consecutive_detections = max(0, self.consecutive_detections - 1)
                
                # Reset fire detected state if no detection for 2 seconds
                if self.fire_detected and time.time() - self.last_detection_time > 2:
                    self.fire_detected = False
            
            # Add status information
            elapsed_time = time.time() - self.start_time
            
            # Status background
            status_bg_height = 60
            cv2.rectangle(display_frame, 
                         (0, display_frame.shape[0] - status_bg_height),
                         (display_frame.shape[1], display_frame.shape[0]),
                         (40, 40, 40), -1)
            
            # Status text
            status_y_base = display_frame.shape[0] - status_bg_height + 20
            
            # Time and detections
            status_text1 = f"Time: {elapsed_time:.1f}s | Detections: {self.detection_count} | FPS: {fps:.1f}"
            cv2.putText(display_frame, status_text1, (10, status_y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Alarm status
            status_text2 = f"Alarm: {'SILENCED' if self.alarm_silenced else 'ACTIVE'}"
            cv2.putText(display_frame, status_text2, (10, status_y_base + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Controls reminder (right aligned)
            controls_text = "Controls: q=Quit, s=Silence, a=Test Alarm"
            text_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            controls_x = display_frame.shape[1] - text_size[0] - 10
            cv2.putText(display_frame, controls_text, (controls_x, status_y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Alert if alarm might be problematic
            if self.fire_detected and not self.alarm_silenced:
                alert_text = "Check if alarm is audible!"
                cv2.putText(display_frame, alert_text, (controls_x, status_y_base + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
            
            # Display the frame
            cv2.imshow('Fire Detection System', display_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                silenced = self.toggle_alarm_silence()
                print(f"Alarm {'silenced' if silenced else 'activated'}")
            elif key == ord('a'):
                # Test alarm
                print("Testing alarm...")
                self.play_alarm()
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


def main():
    # Check if cascade file exists, if not, warn the user
    cascade_path = 'fire_detection.xml'
    if not os.path.exists(cascade_path):
        print(f"Warning: {cascade_path} not found!")
        print("Please download a fire detection Haar cascade XML file and rename it to 'fire_detection.xml'")
        print("or specify the correct path when creating FireDetectionSystem.")
        
        # Provide helpful information
        print("\nTo find a fire detection XML file, you can:")
        print("1. Search online for 'fire detection haar cascade xml'")
        print("2. Or use this direct link to download a sample:")
        print("   https://raw.githubusercontent.com/AndreMuis/FireDetection/master/cascades/fire_detection.xml")
        print("\nDownload the file and place it in the same directory as this script.")
        
        # Ask if the user wants to continue without the file
        while True:
            response = input("\nDo you want to continue without the cascade file? (y/n): ").lower()
            if response == 'n':
                print("Exiting program.")
                sys.exit(1)
            elif response == 'y':
                print("Continuing without cascade file. Fire detection will not work!")
                break
    
    # Create and run the fire detection system
    system = FireDetectionSystem(
        cascade_path=cascade_path,
        camera_index=0,  # Use default camera (change if needed)
        confidence_threshold=0.65,
        save_detections=True
    )
    
    system.run()


if __name__ == "__main__":
    main()











