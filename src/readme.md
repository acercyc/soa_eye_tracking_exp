# Eye Tracking Experiment Suite

This project contains two complementary eye tracking experiments designed to investigate participant preferences for different stimulus types based on movement patterns.

## Experiment 1: Movement Mode Exposure

In this first experiment, there is always a single target object in the scene. The target object has different "movement modes" that fall into two categories:

1. Interactive movement modes: The target object is moved based on the user's gaze (or mouse) position. The primary interactive mode is `locking` mode.
2. Non-interactive movement modes: The target object is moved based on pre-defined algorithms, including `bouncing` and `organic` movement patterns.

After a certain presentation duration, the movement mode changes. This change is accompanied by a change in the appearance of the target object. The experiment alternates between interactive and non-interactive modes (e.g., first mode is interactive, second is non-interactive, third is interactive, etc.).

The experiment ends after a fixed number of trials.

## Experiment 2: Stimulus Preference Assessment

The second experiment builds on the associations formed during Experiment 1. It presents two targets simultaneously (one on the left side and one on the right side of the screen) to assess if participants develop preferences for specific stimuli.

Key features of Experiment 2:
- Each trial presents two targets: an interactive `locking` target and a non-interactive target (`bouncing` or `organic`)
- The non-interactive modes alternate between trials (bouncing → organic → bouncing → etc.)
- Targets are constrained to their respective screen halves (left/right)
- Sounds play when the participant looks at either side continuously for a set duration
- The experiment maintains the same image-sound-movement associations that were established in Experiment 1

The goal is to measure which stimuli participants prefer to look at based on their previous exposure.

## Shared Stimulus Features

### Flash Effect
Both experiments include a target flashing feature that can help capture the subject's attention at the beginning of each trial:

- Enable/disable via `config["stimulus"]["flash"]` setting
- When enabled, the target flashes at the beginning of each trial 
- Each flash consists of a visible phase (0.2s) followed by an invisible phase (0.1s)

### Sound Effect
Sound effects are used in both experiments:

#### Experiment 1:
- Enable/disable via `config["stimulus"]["sound"]["play"]` setting
- Each movement mode is randomly assigned a unique sound file
- Sound plays once at the beginning of each trial
- These sound-movement mode associations are saved for use in Experiment 2

#### Experiment 2:
- Uses the same sound-movement mode associations from Experiment 1
- Sounds play when participants look at either side of the screen for a configurable duration
- Each sound plays only once per trial per side

### Locking Mode: Edge-Dwell Central Cue

When participants look near the screen edge or outside the monitor area for a sustained period in locking mode, a central flashing ring cue appears to guide attention back toward the center. The cue disappears once gaze returns to the safe zone.

- Enabled only for the `locking` movement mode
- Triggered after a dwell period spent near/outside the edge
- Flashing ring draws attention; on/off durations are configurable
- Disable by setting `enabled` to `False`

Configuration lives under:

```python
config["moving_modes"]["locking"]["edge_cue"] = {
    "enabled": True,
    "zone_width": 0.05,  # distance from edge considered near-edge (height units)
    "dwell_seconds": 3.0,  # dwell time near/outside before cue appears
    "ring_radius": 0.06,  # central cue ring radius (height units)
    "ring_color": "white",
    "flash_on": 0.25,  # seconds ring visible per cycle
    "flash_off": 0.25,  # seconds ring hidden per cycle
}
```

Per-frame logs include the following fields for analysis:

- `near_edge`: whether gaze/controller is within the near-edge zone
- `outside_bounds`: whether gaze/controller is outside the screen bounds
- `edge_dwell_s`: seconds accumulated near/outside edge (resets on return)
- `cue_active`: whether the center ring is currently active

Notes:
- `zone_width` is measured in PsychoPy height units; edges are based on the current window aspect and `config["experiment"]["screen_margin"]`.
- The cue is rendered internally by the locking mode and requires no extra calls from the main loop.
- `config["experiment"]["show_pos_indicator"] = True` can help visualize gaze/mouse position while tuning.

## Configuration Parameters

### Experiment 1
- `trial_duration`: Duration of each trial in seconds
- `number_of_trial`: Total number of trials

### Experiment 2
- `N_trials_exp2`: Total number of trials (recommended to be a multiple of 4)
- `effective_trial_duration_exp2`: Duration of each trial, excluding time when gaze is not detected
- `sound_play_gaze_duration`: Duration participant must look at a side to trigger sound

### Moving Modes: Locking Edge Cue

All parameters live under `config["moving_modes"]["locking"]["edge_cue"]`:

- `enabled`: turn the center cue on/off (default: True)
- `zone_width`: near-edge zone width in height units (default: 0.05)
- `dwell_seconds`: delay before cue appears when near/outside (default: 3.0)
- `ring_radius`: cue ring radius in height units (default: 0.06)
- `ring_color`: cue ring color (default: "white")
- `flash_on`: seconds the ring is visible per cycle (default: 0.25)
- `flash_off`: seconds the ring is hidden per cycle (default: 0.25)


# Data Collection and Analysis

## Data Files

Each experiment generates several files per participant:

- `{subjectID}_exp_{date}.csv` or `{subjectID}_exp2_{date}.csv`: Main data log with per-frame information
- `{subjectID}_tobii_{date}.tsv` or `{subjectID}_tobii_exp2_{date}.tsv`: Raw eye-tracking data
- `{subjectID}_config_{date}.json` or `{subjectID}_config_exp2_{date}.json`: Configuration parameters

Experiment 2 uses the date from Experiment 1's files to maintain consistent naming for the same participant.

## Experimental Flow

1. Run Experiment 1 for a participant
2. The configuration (including stimulus mappings) is saved
3. Run Experiment 2 for the same participant, which:
   - Loads the Experiment 1 configuration
   - Uses the same stimulus mappings (mode-to-image and mode-to-sound)
   - Collects preference data with side-by-side presentation

## Analysis Outcomes

The primary outcome measure for Experiment 2 is the proportion of time spent looking at each target, which can reveal:
- Whether participants prefer stimuli associated with specific movement modes from Experiment 1
- If there's a bias toward interactive vs. non-interactive movement modes
- Potential differences in preference between `bouncing` and `organic` non-interactive modes

# Technical Notes

## About sound device

🔊 Audio Playback Requirements (PsychoPy)

To ensure that sound playback works correctly when running this experiment with PsychoPy, please note the following:

#### ✅ Required Package

Make sure you install the correct audio backend package:

```bash
pip install psychopy-sounddevice
```

> **Why this is needed** : PsychoPy uses different audio backends, and the default one (`ptb`, or PsychToolbox) may not work properly on some systems due to hardware or driver conflicts (e.g., unsupported sample rates). Installing `psychopy-sounddevice` enables a more reliable backend (`sounddevice`) that works across most platforms.

#### 🔁 After Installing

After installation, **restart your Python session** or PsychoPy app to allow it to load the new audio backend.
