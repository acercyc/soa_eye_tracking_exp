# Experiment 2: Stimulus Preference Assessment

## 1. Objective

The primary goal of this second experiment is to determine if participants develop a preference for specific stimuli (images and sounds) that were associated with particular movement modes in the first experiment.

## 2. Background

In the first experiment, different movement modes (e.g., locking, bouncing, organic) were paired with unique images and sounds. This repeated exposure might lead participants to form associations and subsequently develop preferences for certain image-sound-movement combinations. This experiment aims to investigate such potential biases.

## 3. Experimental Design

### 3.1. Trial Structure
-   Each trial will present two targets simultaneously: one on the left side of the screen and one on the right. The vertical midline of the screen serves as the boundary between the two sides.
-   One target will always exhibit an "interactive" movement mode (i.e., `locking` mode, controlled by the participant's gaze).
-   The other target will exhibit a "non-interactive" movement mode (e.g., `bouncing` or `organic` mode).
-   The assignment of the interactive and non-interactive modes to the left or right side will be pseudo-randomized across trials.

### 3.2. Stimuli Presentation
-   **Initial Position:** At the beginning of each trial, the two targets will appear at the center of their respective screen halves (e.g., left target at (-0.25 * screen_aspect, 0) and right target at (0.25 * screen_aspect, 0), assuming normalized height coordinates and considering screen aspect ratio).
-   **Movement:** Once presented, both targets will immediately begin their assigned movement patterns strictly within their respective screen halves. The vertical midline of the screen acts as an impenetrable boundary that neither target should cross. Movement parameters for `bouncing` and `organic` modes (e.g., speed, scale) are reused from the general stimulus parameters defined in the Experiment 1 configuration file.
-   **Duration:** Each trial will run for a predefined "effective duration." This duration is specific to Experiment 2 and should be a configurable parameter in the JSON configuration file (e.g., `effective_trial_duration_exp2`), distinct from any `trial_duration` parameter used in Experiment 1. The trial continues and data is collected as long as the participant's gaze is detected on the screen; time when gaze is not detected (e.g., due to blinks or looking away) does not count towards this effective duration.

### 3.3. Movement Modes & Stimuli Mapping
-   **Interactive Mode:** `locking` mode.
-   **Non-Interactive Modes:** `bouncing` mode and `organic` mode.
-   **Configuration Reuse:** Crucially, this experiment **must** use the stimulus mappings (moving mode to image, moving mode to sound) established and saved in the JSON configuration file from the *first experiment* for that specific participant. This ensures consistency in the learned associations being tested.

### 3.4. Sound Playback Logic
-   A sound associated with a target (based on its movement mode and the mapping from Experiment 1's config) will be played **only once per trial** for each target.
-   The sound for a specific target (e.g., the left one) will play if the participant's gaze is detected continuously within that target's half of the screen (left side or right side) for a configurable duration (e.g., `sound_play_gaze_duration` in the configuration, defaulting to 1 second). It is not necessary for the gaze to be precisely on the target's bounding box, only on the correct side of the screen.
-   If the participant looks away from that side and then looks back at the same side (where the same target resides) within the same trial, the sound for that target will **not** play again.
-   This applies independently to both the left and right targets/sides. For example, if the participant looks at the left side of the screen for the `sound_play_gaze_duration`, the left target's sound plays. If they then shift their gaze to the right side of the screen for the `sound_play_gaze_duration`, the right target's sound plays.

### 3.5. Randomization and Trial Sequence
-   **Pairing:** In each trial, the interactive `locking` mode will be paired with one of the two non-interactive modes: `bouncing` or `organic`.

-   **Sequence of Non-Interactive Modes:** The non-interactive mode presented (either `bouncing` or `organic`) will alternate from trial to trial. The sequence will be `bouncing`, then `organic`, then `bouncing`, and so on.
    -   Example for the first 4 trials:
        -   Trial 1: `locking` vs. `bouncing`
        -   Trial 2: `locking` vs. `organic`
        -   Trial 3: `locking` vs. `bouncing`
        -   Trial 4: `locking` vs. `organic`

-   **Side Assignment of Modes:** The assignment of the `locking` mode and the chosen non-interactive mode to the left or right side of the screen is structured to ensure comprehensive balancing across trials. This is achieved using a repeating 4-trial block pattern. The `locking` mode's side and the non-interactive mode's side are determined as follows for each block:
    1.  **Trial 1 of Block:** The non-interactive mode is `bouncing`. `bouncing` is presented on the Left, `locking` is presented on the Right.
    2.  **Trial 2 of Block:** The non-interactive mode is `organic`. `organic` is presented on the Left, `locking` is presented on the Right.
    3.  **Trial 3 of Block:** The non-interactive mode is `bouncing`. `bouncing` is presented on the Right, `locking` is presented on the Left.
    4.  **Trial 4 of Block:** The non-interactive mode is `organic`. `organic` is presented on the Right, `locking` is presented on the Left.
    This 4-trial block repeats for the duration of the experiment.

-   **Overall Distribution and Balancing (for N total trials):**
    -   This sequencing ensures that the non-interactive modes (`bouncing`, `organic`) alternate strictly throughout the experiment.
    -   If N is a multiple of 2 (which it will be if following the alternating non-interactive modes), `locking` will be paired with `bouncing` for N/2 trials and with `organic` for N/2 trials. (e.g., for N=20, there will be 10 trials with `bouncing` and 10 trials with `organic`).
    -   If N is a multiple of 4, this 4-trial block structure ensures perfect balancing:
        -   The `locking` mode appears on the left for N/2 trials and on the right for N/2 trials.
        -   The `bouncing` mode (across its N/2 appearances) appears on the left for N/4 trials and on the right for N/4 trials.
        -   The `organic` mode (across its N/2 appearances) appears on the left for N/4 trials and on the right for N/4 trials.
    -   **Recommendation for N:** It is strongly recommended that the total number of trials N be a multiple of 4 to achieve this complete balancing. If N is even but not a multiple of 4 (e.g., N=22), the sequence will run for `floor(N/4)` full blocks, and the remaining trials (e.g., 2 trials if N mod 4 = 2) will follow the initial part of the block pattern.

## 4. Data Collection

Data logging for Experiment 2 should generally follow the principles and formats used in Experiment 1, but adapted to account for the presence of two stimuli (left and right) simultaneously. For each recorded data frame, the following information should be captured:

-   **Timestamp:** A high-resolution timestamp for each data frame (e.g., `timestamp`).
-   **Trial Number:** The current trial number (e.g., `trial_number`).
-   **Gaze Data:**
    -   Participant's gaze position (x, y coordinates, e.g., `gaze_x`, `gaze_y`).
    -   `NoEyeData` flag: A boolean indicator per data frame, `True` if gaze data is currently invalid or not detected (e.g., during blinks, or if the participant is looking away from the tracker's range), `False` otherwise. This flag is typically provided by the eye-tracking SDK.
-   **Left Target Data:**
    -   Movement mode active (e.g., `left_target_mode`: "locking", "bouncing", or "organic").
    -   Position (x, y coordinates, e.g., `left_target_x`, `left_target_y`).
    -   Associated image file (e.g., `left_target_image`).
    -   Associated sound file (e.g., `left_target_sound`).
-   **Right Target Data:**
    -   Movement mode active (e.g., `right_target_mode`: "locking", "bouncing", or "organic").
    -   Position (x, y coordinates, e.g., `right_target_x`, `right_target_y`).
    -   Associated image file (e.g., `right_target_image`).
    -   Associated sound file (e.g., `right_target_sound`).
-   **Events:** Logged as they occur, with a timestamp. These are typically logged in a separate event file or as special rows/columns in the main data file.
    -   `trial_start`: Timestamp.
    -   `trial_end`: Timestamp.
    -   `sound_play`: Timestamp, target identifier (e.g., "left_target" or "right_target"), and the sound file that was played. Example: `event_type: sound_play, event_timestamp: [timestamp], target: left_target, sound_file: [filename]`.

The data should be structured to clearly associate gaze data with the state of both targets at each point in time.

## 5. Outcome and Analysis

This section outlines the intended future analysis of the collected data to provide context for the data collection requirements; the analysis itself is not part of the experimental procedure's implementation. The primary outcome will be the proportion of "effective time" spent looking at each target (or each side of the screen). This will be analyzed to:
-   Determine if participants spend significantly more time looking at targets associated with specific movement modes from Experiment 1.
-   Assess if there's a preference for stimuli (images/sounds) previously paired with the interactive `locking` mode versus those paired with `bouncing` or `organic` modes.
-   The analysis will compare the dwell time on the side displaying the `locking` mode stimulus versus the side displaying the `non-interactive` mode stimulus, considering the specific images and sounds presented.

## 6. Configuration Parameters

Experiment 2 will require its own set of configuration parameters, typically managed in a JSON file. Some parameters will be specific to Experiment 2, while others, particularly stimulus mappings and general visual properties, will be reused from the participant's Experiment 1 configuration.

### 6.1. Experiment 2 Specific Parameters:
-   `N_trials_exp2` (integer): The total number of trials for Experiment 2. It is strongly recommended that this be a multiple of 4 for complete balancing of conditions.
-   `effective_trial_duration_exp2` (float, seconds): The duration for which data is collected in each trial, excluding time when gaze is not detected.
-   `sound_play_gaze_duration` (float, seconds): The continuous duration the participant's gaze must remain on one side of the screen (left or right) to trigger the sound associated with the target on that side. Default suggestion: 1.0 second.

### 6.2. Reused Parameters from Experiment 1 Configuration:
-   **Stimulus Mappings:** The participant-specific JSON configuration file from Experiment 1, which maps movement modes (`locking`, `bouncing`, `organic`) to specific image files and sound files, **must** be loaded and used.
-   **General Stimulus Parameters:** Parameters like `speed`, `scale` for the stimuli, as defined in the Experiment 1 configuration, should be reused for the `bouncing` and `organic` movement modes in Experiment 2, unless Experiment 2 explicitly defines overrides (though current instructions imply direct reuse).
-   **Screen/Display Parameters:** Any relevant screen resolution or display parameters from Experiment 1's setup should be consistently applied.

The Experiment 2 script should be designed to load the Experiment 1 configuration for a given participant and then apply or override with Experiment 2 specific settings.