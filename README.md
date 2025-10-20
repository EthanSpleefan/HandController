# HandGesture-SoftwareController

## Overview
HandGesture-SoftwareController is a hand gesture-based game controller software designed to provide an intuitive and interactive gaming experience. With this software, users can control games and applications using hand gestures, making it a fun and engaging way to interact with technology.

## Features
- Control games and applications using hand gestures.
- Supports various hand gestures for different actions.
- Easy setup and configuration.
- Compatible with a wide range of games and applications.

## Getting Started
To get started with HandGesture-SoftwareController, follow these steps:

PREPARATION: This program has libraries that only work with Python 3.9. Please ensure you have installed Python 3.9, instructions can be found [here](https://www.python.org/downloads/)

1. **Installation**: Clone this repository to your local machine.

   ```bash
   git clone https://github.com/EthanSpleefan/HandController.git
   ```
2. **Dependencies**: Ensure all libraries listed in [`requirements.txt`](./requirements.txt) are installed on your machine. You can install them with:
   ```bash
   pip install <library>
   ```
   and replace `<library>` with the name of the required library.
   
   or 
   
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the file**: Open `app.py` in your code editor and run the script, or execute:
   ```bash
   python app.py
   ```

## Usage

Once the application is running:
- Press **ESC** to exit the application
- Press **k** to enter keypoint logging mode
- Press **h** to enter point history logging mode
- Press **n** to return to normal mode
- Press **0-9** to select a label number when in logging mode

## Hand Gestures

The application recognizes various hand gestures that can be customized by training the models. Default gestures include:
- Index finger pointing (for navigation)
- Open hand
- Closed fist
- And more (see `model/keypoint_classifier/keypoint_classifier_label.csv`)

## Contributing

Please see [CONTRIBUTORS.md](./CONTRIBUTORS.md) for guidelines on how to contribute to this project.
