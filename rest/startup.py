import os
import subprocess
import sys
import platform

def check_chrome_installation():
    """Check if Chrome is installed and available for image scraping"""
    system = platform.system()
    
    if system == "Windows":
        chrome_paths = [
            os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 'Google\\Chrome\\Application\\chrome.exe'),
            os.path.join(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'), 'Google\\Chrome\\Application\\chrome.exe')
        ]
        
        for path in chrome_paths:
            if os.path.exists(path):
                print(f"Chrome found at: {path}")
                return True
                
        print("Warning: Chrome not found in standard locations. Image scraping may not work.")
        return False
        
    elif system == "Linux":
        # Try to locate Chrome/Chromium on Linux
        try:
            chrome_check = subprocess.run(
                ["which", "google-chrome"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            if chrome_check.returncode == 0:
                chrome_path = chrome_check.stdout.decode().strip()
                print(f"Chrome found at: {chrome_path}")
                return True
                
            # Try chromium as well
            chromium_check = subprocess.run(
                ["which", "chromium-browser"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            if chromium_check.returncode == 0:
                chromium_path = chromium_check.stdout.decode().strip()
                print(f"Chromium found at: {chromium_path}")
                return True
                
            print("Warning: Chrome/Chromium not found. Image scraping may not work properly.")
            return False
            
        except Exception as e:
            print(f"Error checking for Chrome: {e}")
            return False
            
    elif system == "Darwin":  # macOS
        chrome_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            os.path.expanduser("~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        ]
        
        for path in chrome_paths:
            if os.path.exists(path):
                print(f"Chrome found at: {path}")
                return True
                
        print("Warning: Chrome not found in standard locations. Image scraping may not work.")
        return False
        
    else:
        print(f"Unsupported operating system: {system}")
        return False

# When imported, automatically check for Chrome
chrome_available = check_chrome_installation()
