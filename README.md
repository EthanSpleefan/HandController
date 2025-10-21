# HandGesture-SoftwareController

## Overview
HandGesture-SoftwareController is a hand gesture-based game controller software designed to provide an intuitive and interactive gaming experience. With this software, users can control games and applications using hand gestures, making it a fun and engaging way to interact with technology.

## Features
- Control games and applications using hand gestures.
- Supports various hand gestures for different actions.
- Easy setup and configuration.
- Compatible with a wide range of games and applications.

## Getting Started

### Prerequisites

**IMPORTANT**: This program requires Python 3.9 due to specific library dependencies. Ensure you have Python 3.9 installed before proceeding. You can download it from the [official Python website](https://www.python.org/downloads/).

### Installation

1. **Clone the repository** to your local machine:

   ```bash
   git clone https://github.com/EthanSpleefan/HandController.git
   cd HandController
   ```

2. **Install dependencies** using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:

   ```bash
   python app.py
   ```

   Optional command-line arguments:
   ```bash
   python app.py --device 0 --width 960 --height 540 --min_detection_confidence 0.7 --min_tracking_confidence 0.5
   ```

## Usage

Once the application is running:
- Press **ESC** to exit the application
- Press **k** to enter keypoint logging mode
- Press **h** to enter point history logging mode
- Press **n** to return to normal mode
- Press **0-9** to select a label number when in logging mode

### Customizing Gesture Controls

The application uses a configuration file (`gesture_config.json`) to map gestures to keyboard keys. This allows you to customize controls without editing code.

**To modify gesture mappings:**

1. Open `gesture_config.json` in a text editor
2. Add or modify entries in the `gesture_mappings` section:
   ```json
   {
     "gesture_mappings": {
       "1": {
         "key": "right",
         "description": "Right pointing gesture - press right arrow key"
       },
       "2": {
         "key": "up",
         "description": "Pointer gesture - press up arrow key"
       }
     }
   }
   ```
3. The gesture ID (e.g., "1", "2") corresponds to the trained gesture labels in `model/keypoint_classifier/keypoint_classifier_label.csv`
4. The `key` field can be any key supported by the [keyboard library](https://github.com/boppreh/keyboard#api) (e.g., 'a', 'space', 'f1', 'left', 'right')
5. Save the file and restart the application

**To use a different configuration file:**
```bash
python app.py --config my_custom_config.json
```

## Hand Gestures

The application recognizes various hand gestures that can be customized by training the models. Default gestures include:
- **Gesture 0**: Open hand
- **Gesture 1**: Right pointing (triggers right arrow key by default)
- **Gesture 2**: Index finger pointing/pointer (triggers up arrow key by default)
- **Gesture 3**: Left pointing (triggers left arrow key by default)
- And more (see `model/keypoint_classifier/keypoint_classifier_label.csv` for full list)

**Note:** The keyboard actions for each gesture can be customized in `gesture_config.json` without modifying code.

## Project Structure

```
HandController/
├── app.py                          # Main application file
├── requirements.txt                # Python dependencies
├── model/                          # Machine learning models
│   ├── keypoint_classifier/        # Hand gesture classification model
│   └── point_history_classifier/   # Finger movement classification model
├── utils/                          # Utility modules
│   └── cvfpscalc.py               # FPS calculation utility
└── README.md                       # This file
```

## Training Custom Gestures

To train your own gestures:

1. Run the application and press **k** to enter keypoint logging mode
2. Press a number key (0-9) to select the gesture label
3. Perform the gesture in front of the camera to collect training data
4. The data will be saved to `model/keypoint_classifier/keypoint.csv`
5. Train the model using the Jupyter notebooks provided (see `keypoint_classification_EN.ipynb`)

## Technical Details

- **Hand Detection**: MediaPipe Hands for real-time hand landmark detection
- **Gesture Classification**: TensorFlow Lite models for efficient inference
- **FPS**: Optimized for real-time performance with configurable camera settings
- **Keyboard Control**: Automatic keyboard input simulation based on recognized gestures
- **Configuration-based Controls**: JSON configuration file for easy gesture-to-key mapping customization

## Contributing

Please see [CONTRIBUTORS.md](./CONTRIBUTORS.md) for guidelines on how to contribute to this project.
