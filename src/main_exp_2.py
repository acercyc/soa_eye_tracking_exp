from psychopy import visual, core, event
import psychopy_tobii_controller
from abc import ABC, abstractmethod
import numpy as np
import os
import datetime
import json
import GazeStabilizer
import threading
import time

# Experiment 2 specific configuration
config_exp2 = {
    "design_exp2": {
        "number_of_trial_exp2": 20,  # Total number of trials for Experiment 2 (recommended to be multiple of 4)
        "effective_trial_duration_exp2": 10.0,  # Duration per trial (seconds)
        "sound_play_gaze_duration": 0.5  # Duration to trigger sound (seconds)
    }
}

# Import helper functions from Experiment 1
from main import (
    create_data_directory, process_gaze_position, bring_back_to_screen, 
    MovingMode_locking, MovingMode_bouncing, MovingMode_organic, 
    Target_image, Sound_effect, TobiiController, MouseController, break_time
)

class MovingMode_organic_exp2(MovingMode_organic):
    """
    Modified organic movement mode for Experiment 2.
    This class overrides the update method to use the center of the respective screen half
    (left or right) as the attraction point rather than the center of the entire screen.
    """
    def __init__(self, win, is_left_side=True, **kwargs):
        super().__init__(win, **kwargs)
        self.is_left_side = is_left_side
        # Calculate the x-coordinate of the center of the respective half
        screen_aspect = win.aspect
        if self.is_left_side:
            self.center_x = -0.25 * screen_aspect  # Center of left half
        else:
            self.center_x = 0.25 * screen_aspect   # Center of right half
        self.center_y = 0.0  # Vertical center remains the same

    def update(self):
        """
        Updates the dot's position and angle based on its movement dynamics.
        Modified to use the center of the respective screen half as the attraction point.
        """
        # Calculate the vector from the dot to the center of the respective half
        dx = self.x - self.center_x
        dy = self.y - self.center_y
        dist_from_center = np.sqrt(dx**2 + dy**2)  # Distance from the half-center

        # Calculate normalized distance (0 at center, 1 at nearest edge)
        max_dist_ref = min(self.horizontal_limit, self.vertical_limit)
        normalized_dist = min(1.0, dist_from_center / max_dist_ref)

        # Calculate attraction strength proportional to the square of the normalized distance
        current_attraction = self.max_attraction_strength * normalized_dist**2

        # Calculate the angle pointing towards the center of the half
        target_angle = np.arctan2(-dy, -dx)

        # Calculate the shortest angle difference to the target angle
        angle_diff = target_angle - self.angle
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # Add random noise to the angle for wandering behavior
        self.angle += np.random.uniform(-0.5, 0.5) * self.noise_intensity

        # Adjust the angle towards the center using the calculated attraction strength
        self.angle += angle_diff * current_attraction

        # Normalize the angle to keep it within 0 to 2*pi
        self.angle = self.angle % (2 * np.pi)

        # Update the position based on the angle and speed
        delta_x = self.speed * np.cos(self.angle)
        delta_y = self.speed * np.sin(self.angle)
        self.x += delta_x
        self.y += delta_y

        # Handle boundary collisions
        # Reflect the dot's movement if it hits the horizontal boundaries
        if self.x + self.radius > self.horizontal_limit:
            self.x = self.horizontal_limit - self.radius
            self.angle = np.pi - self.angle
        elif self.x - self.radius < -self.horizontal_limit:
            self.x = -self.horizontal_limit + self.radius
            self.angle = np.pi - self.angle

        # Reflect the dot's movement if it hits the vertical boundaries
        if self.y + self.radius > self.vertical_limit:
            self.y = self.vertical_limit - self.radius
            self.angle = -self.angle
        elif self.y - self.radius < -self.vertical_limit:
            self.y = -self.vertical_limit + self.radius
            self.angle = -self.angle

        # Normalize the angle again after reflection
        self.angle = self.angle % (2 * np.pi)

        # Update the pos property for consistency with MovingMode interface
        self.pos = np.array([self.x, self.y])

class Design_Exp2:
    """
    Design class specific to Experiment 2.
    Manages the trial sequence, pairing of modes, and side assignment.
    """
    def __init__(self, number_of_trials=20):
        self.number_of_trials = number_of_trials
        self.trial_sequences = []  # Will store the sequence of trials
        self.left_side_mode = []  # Will store the mode on the left side for each trial
        self.right_side_mode = []  # Will store the mode on the right side for each trial
        
    def generate_design(self):
        """
        Generate the design for Experiment 2 with the specific sequencing requirements.
        - Alternate non-interactive modes (bouncing, organic)
        - Structure the side assignment with the 4-trial block pattern
        """
        self.trial_sequences = []
        self.left_side_mode = []
        self.right_side_mode = []
        
        # Number of complete 4-trial blocks
        num_blocks = self.number_of_trials // 4
        remaining_trials = self.number_of_trials % 4
        
        # Generate the full blocks
        for block in range(num_blocks):
            # Trial 1: bouncing on left, locking on right
            self.left_side_mode.append('bouncing')
            self.right_side_mode.append('locking')
            
            # Trial 2: organic on left, locking on right
            self.left_side_mode.append('organic')
            self.right_side_mode.append('locking')
            
            # Trial 3: locking on left, bouncing on right
            self.left_side_mode.append('locking')
            self.right_side_mode.append('bouncing')
            
            # Trial 4: locking on left, organic on right
            self.left_side_mode.append('locking')
            self.right_side_mode.append('organic')
        
        # Handle any remaining trials (if N is not a multiple of 4)
        if remaining_trials > 0:
            for i in range(remaining_trials):
                if i == 0:
                    # First remaining trial: bouncing on left, locking on right
                    self.left_side_mode.append('bouncing')
                    self.right_side_mode.append('locking')
                elif i == 1:
                    # Second remaining trial: organic on left, locking on right
                    self.left_side_mode.append('organic')
                    self.right_side_mode.append('locking')
                elif i == 2:
                    # Third remaining trial: locking on left, bouncing on right
                    self.left_side_mode.append('locking')
                    self.right_side_mode.append('bouncing')
        
        # Create trial sequences
        for i in range(self.number_of_trials):
            self.trial_sequences.append({
                'left_mode': self.left_side_mode[i],
                'right_mode': self.right_side_mode[i]
            })
        
        return self.trial_sequences
    
    def load_exp1_config(self, config_path):
        """
        Load the configuration from Experiment 1 for a specific participant.
        
        Parameters:
            config_path: Path to the Experiment 1 configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config_exp1 = json.load(f)
            print(f"Loaded Experiment 1 configuration from: {config_path}")
            return config_exp1
        except Exception as e:
            print(f"Error loading Experiment 1 configuration: {e}")
            return None

class DataManager_Exp2:
    """
    Manages data logging specific to Experiment 2.
    """
    def __init__(self, data_folder):
        self.data_path_exp = None
        self.data_path_tobii = None
        self.data_path_config = None
        self.data_folder = data_folder
        self.date = datetime.datetime.today().strftime("%Y%m%d%H%M")
        self.iDataEntry = 0  # Initialize iEntry to 0
        self.file = None

    def enter_subj_id_and_exp1_config(self):
        """
        Input subjectID from CLI and specify the Experiment 1 config file
        """
        subjectID = input("Enter subject ID: ")
        if not subjectID:
            raise ValueError("Subject ID cannot be empty.")
        
        # Get the list of available config files for this subject
        config_files = []
        for file in os.listdir(self.data_folder):
            if file.startswith(f"{subjectID}_config_") and file.endswith(".json"):
                config_files.append(file)
        
        # Let the user select the config file
        if not config_files:
            raise ValueError(f"No Experiment 1 configuration files found for subject {subjectID}")
        
        print("Available Experiment 1 configuration files:")
        for i, file in enumerate(config_files):
            print(f"{i+1}: {file}")
        
        selection = int(input("Select the configuration file number: ")) - 1
        if selection < 0 or selection >= len(config_files):
            raise ValueError("Invalid selection")
        exp1_config_path = os.path.join(self.data_folder, config_files[selection])
        
        # Extract date from the Experiment 1 config filename
        # Format is: "{subjectID}_config_YYYYMMDDHHMM.json"
        exp1_date = config_files[selection].split('_config_')[1].replace('.json', '')
        
        # Set up data paths for Experiment 2 with the same date as Experiment 1
        self.data_path_exp = os.path.join(
            self.data_folder, f"{subjectID}_exp2_{exp1_date}.csv")
        self.data_path_tobii = os.path.join(
            self.data_folder, f"{subjectID}_tobii_exp2_{exp1_date}.tsv")
        self.data_path_config = os.path.join(
            self.data_folder, f"{subjectID}_config_exp2_{exp1_date}.json")
        print(
            f"Data paths set: {self.data_path_exp}, {self.data_path_tobii}, {self.data_path_config}")
        
        return exp1_config_path

    def save_config(self, config):
        """
        Save the configuration to a JSON file.
        """
        try:
            with open(self.data_path_config, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Configuration saved to: {self.data_path_config}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def log_data(self, frame_data):
        """
        Log frame data directly to the experiment data file.
        frame_data: dict containing data to log

        This method streams data directly to a file without keeping everything in memory.
        """
        # Open the file if not already open
        if self.file is None:
            try:
                # Open file for writing
                self.file = open(self.data_path_exp, 'w')
                print(f"Opened data file for writing: {self.data_path_exp}")
            except Exception as e:
                print(f"Error opening data file: {e}")
                return

        # Write header if this is the first entry (iDataEntry == 0)
        if self.iDataEntry == 0:
            # Get column names from the dictionary keys
            header = ','.join(frame_data.keys())
            self.file.write(header + '\n')

        # Convert values to strings and write to file
        values = []
        for value in frame_data.values():
            if isinstance(value, str):
                # Escape commas in strings by wrapping in quotes
                values.append(f'"{value}"')
            else:
                values.append(str(value))

        # Write the values as a CSV row
        row = ','.join(values)
        self.file.write(row + '\n')

        # Flush buffer to ensure data is written immediately
        self.file.flush()

        # Increment entry counter
        self.iDataEntry += 1

    def close_file(self):
        """
        Close the data file.
        """
        if self.file is not None:
            self.file.close()
            print(f"Data file closed: {self.data_path_exp}")
            self.file = None

def run_experiment_2(controller_type='tobii'):
    """
    Main function to run Experiment 2.
    """
    # Create data directory if it doesn't exist
    data_folder = create_data_directory()
    
    # Initialize data manager for Experiment 2
    data_manager = DataManager_Exp2(data_folder)
    
    # Get subject ID and path to Experiment 1 configuration
    exp1_config_path = data_manager.enter_subj_id_and_exp1_config()
      # Load Experiment 1 configuration
    design = Design_Exp2(config_exp2["design_exp2"]["number_of_trial_exp2"])
    config_exp1 = design.load_exp1_config(exp1_config_path)
    
    # Merge configurations (Exp1 + Exp2 specific parameters)
    config_combined = config_exp1.copy()
    config_combined.update({"design_exp2": config_exp2["design_exp2"]})
    
    # Save the combined configuration
    data_manager.save_config(config_combined)
    
    # Generate trial sequences
    trials = design.generate_design()
    
    # Create a psychopy window
    win = visual.Window(units='height',
                        monitor='default',
                        fullscr=config_exp1["experiment"]["fullscreen"],
                        colorSpace='rgb255',
                        color=(100, 100, 100))
    
    # Hide the mouse if in fullscreen mode
    if config_exp1["experiment"]["fullscreen"]:
        win.mouseVisible = False
        event.Mouse(visible=False)
    
    # Initialize text stimulus for instructions
    instructions = visual.TextStim(
        win,
        text="Press 'escape' to exit",
        pos=(0, 0),
        color='white',
        height=0.05
    )
    
    # Initialize controller (eye tracker or mouse)
    if controller_type == 'tobii':
        controller = TobiiController(
            win=win, stabilizer_type=config_exp1["tobii"]["stabilizer_type"])
    else:
        controller = MouseController(win=win)
    
    # Initialize position indicator if enabled
    pos_indicator = None
    if config_exp1["experiment"]["show_pos_indicator"]:
        pos_indicator = visual.Circle(
            win,
            radius=0.005,  # Small radius for the indicator
            fillColor='red',
            lineColor='red',
            opacity=0.7,  # Semi-transparent
            autoLog=False  # Disable logging for performance
        )
    
    # Initialize left and right targets
    left_target = Target_image(win, scale=config_exp1["stimulus"]["scale"])
    right_target = Target_image(win, scale=config_exp1["stimulus"]["scale"])
      # Initialize moving modes - we'll create instances as needed during trials
    locking_mode_left = MovingMode_locking(win)
    locking_mode_right = MovingMode_locking(win)
    bouncing_mode_left = MovingMode_bouncing(win, speed=config_exp1["stimulus"]["speed"])
    bouncing_mode_right = MovingMode_bouncing(win, speed=config_exp1["stimulus"]["speed"])
    organic_mode_left = MovingMode_organic_exp2(win, is_left_side=True, speed=config_exp1["stimulus"]["speed"])
    organic_mode_right = MovingMode_organic_exp2(win, is_left_side=False, speed=config_exp1["stimulus"]["speed"])
    
    # Define movement constraints (screen halves)
    screen_aspect = win.aspect
    left_horizontal_limit = 0.5 * screen_aspect * -1  # Left edge
    right_horizontal_limit = 0.5 * screen_aspect  # Right edge
    midline = 0.0  # Vertical midline
    vertical_limit_top = 0.5 - config_exp1["experiment"]["screen_margin"]
    vertical_limit_bottom = -0.5 + config_exp1["experiment"]["screen_margin"]
    
    # Calculate initial positions (center of each half)
    left_initial_pos = np.array([-0.25 * screen_aspect, 0.0])
    right_initial_pos = np.array([0.25 * screen_aspect, 0.0])
    
    # Initialize sound effects
    left_sound_effect = None
    right_sound_effect = None
    if config_exp1["stimulus"]["sound"]["play"]:
        # Create mappings for left and right targets (will be set per trial)
        left_sound_effect = Sound_effect({})
        right_sound_effect = Sound_effect({})
    
    # Start the eye tracking session if using Tobii
    if controller_type == 'tobii':
        controller.subscribe(data_manager.data_path_tobii)
    
    # Record experiment start event
    controller.record_event("Experiment 2 started")
    
    # Show welcome message
    instructions.setText("Welcome to Experiment 2!\nPress 'space' to start")
    instructions.draw()
    win.flip()
    event.waitKeys(keyList=['space'])
    
    win.flip()
    core.wait(2)
    
    # Run trials
    for iTrial, trial in enumerate(trials):
        # Record trial start event
        controller.record_event(f"Exp2 Trial {iTrial+1} started")
        
        # Get the movement modes for this trial
        left_mode = trial['left_mode']
        right_mode = trial['right_mode']
        
        # Set up the left target
        if left_mode == 'locking':
            current_left_mode = locking_mode_left
        elif left_mode == 'bouncing':
            current_left_mode = bouncing_mode_left
        elif left_mode == 'organic':
            current_left_mode = organic_mode_left
        
        # Set up the right target
        if right_mode == 'locking':
            current_right_mode = locking_mode_right
        elif right_mode == 'bouncing':
            current_right_mode = bouncing_mode_right
        elif right_mode == 'organic':
            current_right_mode = organic_mode_right
        
        # Reset the modes to their initial positions
        current_left_mode.reset(left_initial_pos)
        current_right_mode.reset(right_initial_pos)
        
        # Set the appropriate images for the targets based on mode-to-image mapping from Exp1
        left_image = config_exp1["runtime"]["moving_mode_to_image"][left_mode]
        right_image = config_exp1["runtime"]["moving_mode_to_image"][right_mode]
        left_target.set_stim(left_image)
        right_target.set_stim(right_image)
        
        # Set up sound effects
        if config_exp1["stimulus"]["sound"]["play"]:
            left_sound_mapping = {left_mode: config_exp1["runtime"]["moving_mode_to_sound"][left_mode]}
            right_sound_mapping = {right_mode: config_exp1["runtime"]["moving_mode_to_sound"][right_mode]}
            left_sound_effect = Sound_effect(left_sound_mapping)
            right_sound_effect = Sound_effect(right_sound_mapping)
        
        # Reset sound played flags
        left_sound_played = False
        right_sound_played = False
        
        # Track time looking at each side for sound triggering
        left_side_gaze_time = 0.0
        right_side_gaze_time = 0.0
        looking_at_left = False
        looking_at_right = False
        
        # Initialize gaze side variables
        on_left_side = False
        on_right_side = False
        
        # Flash targets if enabled
        if config_exp1["stimulus"]["flash"]:
            # Flash left target
            for i in range(3):
                left_target.set_pos(current_left_mode.pos)
                right_target.set_pos(current_right_mode.pos)
                left_target.draw()
                right_target.draw()
                win.flip()
                core.wait(0.2)
                win.flip()
                core.wait(0.1)
        
        # Start trial loop
        start_time = core.getTime()
        effective_time = 0.0
        frame_num = 0
        last_time = start_time
        
        while effective_time < config_exp2["design_exp2"]["effective_trial_duration_exp2"]:
            # Get current time
            current_time = core.getTime()
            dt = current_time - last_time
            last_time = current_time
            
            # Get gaze position
            controller_pos = controller.get_pos()
            
            # Check if gaze is valid (not NoEyeData)
            if not controller.isNoData:
                effective_time += dt
                
                # Determine which side the gaze is on
                on_left_side = controller_pos[0] < midline
                on_right_side = controller_pos[0] > midline
                
                # Update gaze side timers for sound triggering
                if on_left_side:
                    if not looking_at_left:  # Just started looking at left
                        looking_at_left = True
                        looking_at_right = False
                        left_side_gaze_time = 0.0  # Reset timer
                    else:  # Continue looking at left
                        left_side_gaze_time += dt
                        
                    # Check if sound should play
                    if (left_side_gaze_time >= config_exp2["design_exp2"]["sound_play_gaze_duration"] and 
                        not left_sound_played and config_exp1["stimulus"]["sound"]["play"]):
                        left_sound_effect.play(left_mode)
                        left_sound_played = True
                        controller.record_event(f"Sound played for left side ({left_mode})")
                        
                elif on_right_side:
                    if not looking_at_right:  # Just started looking at right
                        looking_at_right = True
                        looking_at_left = False
                        right_side_gaze_time = 0.0  # Reset timer
                    else:  # Continue looking at right
                        right_side_gaze_time += dt
                        
                    # Check if sound should play
                    if (right_side_gaze_time >= config_exp2["design_exp2"]["sound_play_gaze_duration"] and 
                        not right_sound_played and config_exp1["stimulus"]["sound"]["play"]):
                        right_sound_effect.play(right_mode)
                        right_sound_played = True
                        controller.record_event(f"Sound played for right side ({right_mode})")
            
            # Update left target position based on its mode
            if left_mode == 'locking' and on_left_side:
                # Only update if gaze is on the left side
                constrained_pos = controller_pos.copy()
                # Ensure target stays in left half
                constrained_pos[0] = max(constrained_pos[0], left_horizontal_limit)
                constrained_pos[0] = min(constrained_pos[0], midline - config_exp1["stimulus"]["scale"])
                # Update y within vertical limits
                constrained_pos[1] = max(constrained_pos[1], vertical_limit_bottom)
                constrained_pos[1] = min(constrained_pos[1], vertical_limit_top)
                current_left_mode.update(constrained_pos)
            elif left_mode != 'locking':
                # For non-interactive modes, update normally but constrain to left half
                current_left_mode.update()
                # Ensure left target stays in left half
                if current_left_mode.pos[0] > midline:
                    current_left_mode.pos[0] = midline - 0.01  # Small offset from midline
                    # Reverse x direction for bouncing mode
                    if left_mode == 'bouncing':
                        current_left_mode.velocity[0] = -abs(current_left_mode.velocity[0])
            
            # Update right target position based on its mode
            if right_mode == 'locking' and on_right_side:
                # Only update if gaze is on the right side
                constrained_pos = controller_pos.copy()
                # Ensure target stays in right half
                constrained_pos[0] = min(constrained_pos[0], right_horizontal_limit)
                constrained_pos[0] = max(constrained_pos[0], midline + config_exp1["stimulus"]["scale"])
                # Update y within vertical limits
                constrained_pos[1] = max(constrained_pos[1], vertical_limit_bottom)
                constrained_pos[1] = min(constrained_pos[1], vertical_limit_top)
                current_right_mode.update(constrained_pos)
            elif right_mode != 'locking':
                # For non-interactive modes, update normally but constrain to right half
                current_right_mode.update()
                # Ensure right target stays in right half
                if current_right_mode.pos[0] < midline:
                    current_right_mode.pos[0] = midline + 0.01  # Small offset from midline
                    # Reverse x direction for bouncing mode
                    if right_mode == 'bouncing':
                        current_right_mode.velocity[0] = abs(current_right_mode.velocity[0])
            
            # Update target positions
            left_target.set_pos(current_left_mode.pos)
            right_target.set_pos(current_right_mode.pos)
            
            # Draw targets
            left_target.draw()
            right_target.draw()
            
            # Draw position indicator if enabled
            if config_exp1["experiment"]["show_pos_indicator"]:
                pos_indicator.setPos(controller_pos)
                pos_indicator.draw()
            
            # Check for escape key
            keys = event.getKeys()
            if 'escape' in keys:
                controller.record_event("Experiment 2 interrupted by user")
                data_manager.close_file()
                if controller_type == 'tobii':
                    controller.unsubscribe()
                win.close()
                core.quit()
                return
            
            # Log data for this frame
            frame_data = {
                'timestamp': current_time,
                'trial_number': iTrial + 1,
                'frame_num': frame_num,
                'time_trial': current_time - start_time,
                'effective_time': effective_time,
                'gaze_x': controller_pos[0],
                'gaze_y': controller_pos[1],
                'NoEyeData': controller.isNoData if controller_type == 'tobii' else False,
                'left_target_mode': left_mode,
                'left_target_x': current_left_mode.pos[0],
                'left_target_y': current_left_mode.pos[1],
                'left_target_image': left_image,
                'left_target_sound': config_exp1["runtime"]["moving_mode_to_sound"][left_mode],
                'left_sound_played': left_sound_played if config_exp1["stimulus"]["sound"]["play"] else None,
                'right_target_mode': right_mode,
                'right_target_x': current_right_mode.pos[0],
                'right_target_y': current_right_mode.pos[1],
                'right_target_image': right_image,
                'right_target_sound': config_exp1["runtime"]["moving_mode_to_sound"][right_mode],
                'right_sound_played': right_sound_played if config_exp1["stimulus"]["sound"]["play"] else None,
                'looking_at_left': looking_at_left,
                'looking_at_right': looking_at_right,
                'left_side_gaze_time': left_side_gaze_time,
                'right_side_gaze_time': right_side_gaze_time
            }
            data_manager.log_data(frame_data)
            
            frame_num += 1
            win.flip()
        
        # Record trial end event
        controller.record_event(f"Exp2 Trial {iTrial+1} ended")
        
        # Break time handling
        # if this is the final trial, skip the break
        if iTrial != len(trials) - 1:
            if config_exp1["experiment"]["break"]["enabled"] and (iTrial + 1) % config_exp1["experiment"]["break"]["every_n_trials"] == 0:
                # Calculate break duration (in seconds)
                break_duration = config_exp1["experiment"]["break"]["duration"]

                # Get the movie path for the break
                movie_path = config_exp1["experiment"]["break"]["movie"][0]

                # Run the break time function
                break_time(win, movie_path, break_duration)        
        
        # Brief pause between trials
        if iTrial < len(trials) - 1:  # Not after the last trial
            core.wait(0.5)  # Half-second pause between trials
    
    # Show completion message
    instructions.setText("Experiment 2 completed!\nPress 'escape' to exit")
    instructions.draw()
    win.flip()
    
    # Wait for escape key to exit
    event.waitKeys(keyList=['escape'])
    
    # Record experiment end event
    controller.record_event("Experiment 2 completed")
    
    # Clean up
    data_manager.close_file()
    if controller_type == 'tobii':
        controller.unsubscribe()
    win.close()
    core.quit()

if __name__ == "__main__":
    # Initialize sound devices if sound is enabled
    if config_exp2.get("stimulus", {}).get("sound", {}).get("play", True):
        from psychopy import prefs
        prefs.hardware['audioLib'] = ['pygame']
        prefs.hardware['audioSampleRate'] = 44100
        from psychopy import sound
    
    run_experiment_2()