# Eye Tracking Experiment Suite

This project implements a two-stage eye tracking experiment suite designed to investigate how users interact with and develop preferences for visual targets using different movement patterns.

## Overview

The suite consists of two complementary experiments:

1. **Experiment 1**: Presents a single visual target that moves according to different algorithms ("moving modes"). Each movement mode is paired with specific visual and auditory stimuli.
2. **Experiment 2**: Presents two targets simultaneously (left and right sides) to assess preferences developed during the first experiment.

## Features

### Movement Modes

Three different target movement modes are used in both experiments:

- **Locking**: Target locks onto the user's gaze or mouse position (interactive mode)
- **Bouncing**: Target moves in straight lines and bounces off screen edges (non-interactive mode)
- **Organic**: Target moves with naturalistic, slightly random movements (non-interactive mode)

### Technical Features

- Integration with Tobii eye tracker with stabilization algorithms
- Mouse fallback option for debugging or when eye tracker is unavailable
- Support for both image and circle stimuli
- Configurable trial sequence and experiment parameters
- Visual position indicator showing the current gaze/mouse position
- Sound effects associated with different movement patterns
- Comprehensive data logging and event recording

### Experiment 2 Specific Features

- Simultaneous presentation of two targets (left and right sides)
- Structured trial sequencing to balance movement modes across sides
- Sound playback triggered by continuous gaze on a particular side
- Maintains stimulus mappings (image and sound) from Experiment 1

## Requirements

- Python 3.6+
- PsychoPy
- Tobii Eye Tracker and Tobii Research SDK
- psychopy_tobii_controller

## Configuration

### Experiment 1 Configuration

Experiment 1 uses the following configuration parameters:

```python
config = {
    "design": {
        "trial_duration": 10,  # in seconds
        "number_of_trial": 20
    },
    "experiment": {
        "fullscreen": True,
        "show_pos_indicator": False,  # Show position indicator on the screen
        "screen_margin": 0.02,  # Margin from the screen edge in height units
    },
    "controller": {
        "type": "tobii",  # Options: 'mouse' or 'tobii'
    },
    "tobii": {
        "calibration": False,  # Whether to perform calibration
        "calibration_points": 5,  # Number of calibration points (5 or 9)
        "stabilizer_type": "moving_average",  # Stabilizer type: 'none', 'moving_average'
    },
    "stimulus": {
        "target_type": "image",  # Options: 'circle' or 'image'
        "scale": 0.05,  # Scale of the target in height units
        "speed": 0.005,  # Speed of the target in bouncing/organic modes
        "flash": True,  # Whether to flash the target at the beginning of each trial
        "sound": {
            "play": True  # Whether to play sound when the target appears
        },
        "images": ["img1.png", "img2.png", "img3.png"],  # Image files for the target
        "sounds": ["sound1.wav", "sound2.wav", "sound3.wav"]  # Sound files for the target
    }
}
```

### Experiment 2 Configuration

Experiment 2 extends the configuration from Experiment 1 with additional parameters:

```python
config_exp2 = {
    "design_exp2": {
        "N_trials_exp2": 20,  # Total number of trials (recommended to be multiple of 4)
        "effective_trial_duration_exp2": 4.0,  # Duration per trial (seconds)
        "sound_play_gaze_duration": 1.0  # Duration to trigger sound (seconds)
    }
}
```

## Usage

### Running Experiment 1

1. Ensure all dependencies are installed and the Tobii eye tracker is connected (if using)
2. Place image stimuli in the `./images` directory if using image targets
3. Place sound files in the `./sounds` directory if using sound effects
4. Run the Experiment 1 script:

```
python src/main.py
```

5. Enter the subject ID when prompted
6. If using the eye tracker, follow the calibration procedure if enabled
7. The experiment will begin after pressing the spacebar
8. Press 'Escape' to exit the experiment at any time

### Running Experiment 2

1. Ensure that the participant has already completed Experiment 1
2. Run the Experiment 2 script:

```
python src/main_exp_2.py
```

3. Enter the same subject ID as used in Experiment 1
4. Select the corresponding configuration file from Experiment 1 when prompted
5. The experiment will begin after pressing the spacebar
6. Press 'Escape' to exit the experiment at any time

## Data

Data is stored in the `./data` directory (created automatically if it doesn't exist):

- Experiment 1 files: `{subjectID}_exp_{date}.csv`, `{subjectID}_tobii_{date}.tsv`, `{subjectID}_config_{date}.json`
- Experiment 2 files: `{subjectID}_exp2_{date}.csv`, `{subjectID}_tobii_exp2_{date}.tsv`, `{subjectID}_config_exp2_{date}.json`

Experiment 2 uses the same date identifier as Experiment 1 to maintain consistent naming across both experiments for the same participant.

## Project Structure

- `src/main.py`: Experiment 1 main script
- `src/main_exp_2.py`: Experiment 2 main script
- `src/images/`: Directory for image stimuli
- `src/sounds/`: Directory for sound stimuli
- `src/GazeStabilizer.py`: Eye-tracking data stabilization algorithms
- `data/`: Directory for experimental data
- `.github/instructions/`: Contains detailed experiment specifications

## Classes

The experiments use an object-oriented approach with the following main classes:

### Shared Classes

- `MovingMode`: Abstract base class for different target movement patterns
  - `MovingMode_bouncing`: Implements bouncing movement
  - `MovingMode_organic`: Implements organic, naturalistic movement
  - `MovingMode_locking`: Implements gaze/mouse-following behavior
- `Target`: Abstract base class for visual targets
  - `Target_circle`: Simple circular target
  - `Target_image`: Image-based target
- `Controller`: Abstract base class for input controllers
  - `TobiiController`: Handles eye tracker input
  - `MouseController`: Handles mouse input (fallback option)
- `Sound_effect`: Manages sound playback

### Experiment 1 Classes

- `Design`: Manages experiment design and trial sequence
- `DataManager`: Handles data logging and file management

### Experiment 2 Classes

- `Design_Exp2`: Manages trial sequence with balanced side assignments
- `DataManager_Exp2`: Handles data logging with references to Experiment 1 files
