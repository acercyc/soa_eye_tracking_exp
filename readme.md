# Eye Tracking Experiment

This project implements an interactive eye tracking experiment that investigates how users track and interact with visual targets using different movement patterns.

## Overview

The experiment presents a visual target that moves according to different algorithms ("moving modes"). The target can be controlled either by eye movements (tracked with a Tobii eye tracker) or by mouse movements.

## Features

- Three different target movement modes:

  - **Bouncing**: Target moves in straight lines and bounces off screen edges
  - **Organic**: Target moves with naturalistic, slightly random movements with a tendency to return to center
  - **Locking**: Target locks onto the user's gaze or mouse position when it comes close enough
- Support for both image and circle stimuli
- Integration with Tobii eye tracker
- Mouse fallback option for debugging or when eye tracker is unavailable
- Visual position indicator showing the current gaze/mouse position
- Configurable trial sequence and experiment parameters

## Requirements

- Python 3.6+
- PsychoPy
- Tobii Eye Tracker and Tobii Research SDK
- psychopy_tobii_controller

## Configuration

The experiment is configured through the `para` dictionary in the main script:

```python
config = {
    "design": {
        "presentation_duration": 10,  # in seconds
        "number_of_presentations": 10
    },
    "experiment": {
        "fullscreen": True,
        "show_pos_indicator": True,  # Show position indicator on the screen
    },
    "controller": {
        "type": "mouse",  # Options: 'mouse' or 'tobii'
    },
    "tobii": {
        "calibration": True,  # Whether to perform calibration
        "calibration_points": 5,  # Number of calibration points (5 or 9)
    },
    "stimulus": {
        "speed": 0.005,  # Speed of the target in bouncing mode
    },
}
```

## Usage

1. Ensure all dependencies are installed and the Tobii eye tracker is connected (if using)
2. Place image stimuli in the `./images` directory if using image targets
3. Run the main script:

```
python src/main.py
```

4. If using the eye tracker, follow the calibration procedure
5. The experiment will begin after pressing the spacebar
6. Press 'Escape' to exit the experiment at any time

## Data

Data is stored in the `../data` directory (created automatically if it doesn't exist).

## Project Structure

- `src/main.py`: Main experiment script
- `src/images/`: Directory for image stimuli
- `data/`: Directory for experimental data
- `Spatial distortion/`: Contains related spatial distortion experiments

## Classes

The experiment uses an object-oriented approach with the following main classes:

- `MovingMode`: Abstract base class for different target movement patterns

  - `MovingMode_bouncing`: Implements bouncing movement
  - `MovingMode_organic`: Implements organic, naturalistic movement
  - `MovingMode_locking`: Implements gaze/mouse-following behavior
- `Target`: Abstract base class for visual targets

  - `Target_circle`: Simple circular target
  - `Target_image`: Image-based target
- `Controller`: Handles input from eye tracker or mouse
- `Design`: Manages experiment design and trial sequence
