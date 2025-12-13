import subprocess
import webbrowser
import time
import threading

def open_browser():
    time.sleep(3)
    webbrowser.open('http://localhost:8501')

def run_streamlit():
    print("Starting Crop Yield Prediction Dashboard...")
    print("Opening at: http://localhost:8501")
    
    # Start browser in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run streamlit
    subprocess.run([
        'streamlit', 'run', 'dashboard/app.py',
        '--server.port', '8501',
        '--server.address', 'localhost'
    ])

if __name__ == "__main__":
    run_streamlit()