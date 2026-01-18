# 🚦 Smart City Traffic Analytics System

A professional computer vision system for real-time traffic monitoring, flow analysis, and data visualization.

## Features

- **Real-Time Detection**: Uses YOLOv8 for accurate vehicle detection.
- **Object Tracking**: Implements tracking to assign unique IDs to vehicles across frames.
- **Flow Counting**: Accurately counts vehicles crossing a virtual line (Vehicles Per Minute).
- **Data Logging**: Automatically saves traffic stats to a local SQLite database.
- **Analytics Dashboard**: Interactive Streamlit dashboard to visualize traffic trends and peak hours.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Run the Detection System
This will open the video feed, track vehicles, and log data.
```bash
python main.py
```
*Press 'q' to stop the system.*

### 2. Launch the Analytics Dashboard
Open a new terminal and run:
```bash
streamlit run dashboard.py
```
This will open a web page showing real-time charts and statistics.

## Project Structure

- `main.py`: Main application entry point.
- `dashboard.py`: Streamlit data visualization.
- `src/`:
    - `detector.py`: YOLO model logic.
    - `tracker.py`: Tracking and counting logic.
    - `database.py`: Data storage.
    - `config.py`: Settings (paths, classes, thresholds).
