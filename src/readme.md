# Experiment Design

In this experiment, there is always a target object in the scene. The target object has different "movement modes." There are two categories of movement modes.

1. Interactive movement modes: The target object is moved based on the user's action. It could be eye movement, mouse movement, and so on.
2. Non-interactive movement modes: The target object is moved based on a pre-defined algorithm.

After a certain presentation duration, the movement modes change. This change could be accompanied by a change in appearance of the target object. The change should interleave with the two categories of movement modes. For example, the first mode is interactive, and the second mode is non-interactive. The third mode is interactive, and the fourth mode is non-interactive.

After a fixed number of changes, the experiment ends.

## Stimulus Features

### Flash Effect
The experiment includes a target flashing feature that can help capture the subject's attention at the beginning of each trial:

- Enable/disable via `config["stimulus"]["flash"]` setting
- When enabled, the target flashes 5 times (default) at the beginning of each trial 
- Each flash consists of a visible phase (0.2s) followed by an invisible phase (0.1s)

### Sound Effect
Sound effects can be assigned to different movement modes:

- Enable/disable via `config["stimulus"]["sound"]["play"]` setting
- When enabled, each movement mode is randomly assigned a unique sound file
- Sound plays once at the beginning of each trial
- Sounds are preloaded to improve performance


# Notes

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
