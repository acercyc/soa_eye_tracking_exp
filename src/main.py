import tobii_research as tr
from psychopy import visual, core, event
import psychopy_tobii_controller
from abc import ABC, abstractmethod
import numpy as np
import os
import datetime
import json

config = {
    "design": {
        "trial_duration": 10,  # in seconds
        "number_of_trial": 20
    },
    "experiment": {
        "fullscreen": True,
        "show_pos_indicator": False,  # Show position indicator on the screen
    },
    "controller": {
        "type": "tobii",  # Options: 'mouse' or 'tobii'
    },
    "tobii": {
        "calibration": True,  # Whether to perform calibration
        "calibration_points": 5,  # Number of calibration points (5 or 9)
    },
    "stimulus": {
        "speed": 0.005,  # Speed of the target in bouncing mode
    },

}


# create data directory if not exist
data_folder = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../data'))
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    print(f"Data directory created: {data_folder}")
else:
    print(f"Data directory already exists: {data_folder}")


# Get all image file paths in the /image directory
def get_images():
    """
    Retrieves all image file paths from the './images' directory.

    Returns:
        list: A list of file paths to the images.
    """
    image_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), './images'))
    if os.path.exists(image_dir):
        images = [os.path.join(image_dir, f) for f in os.listdir(
            image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        print(f"Found {len(images)} images in the directory.")
    else:
        print(f"Image directory not found: {image_dir}")
        images = []
    return images


def get_colors(n=10):
    """
    Generates a list of visually distinct colors.

    Parameters:
        n (int): Number of colors to generate (default: 10)

    Returns:
        list: A list of color strings in hex format (e.g., '#FF0000' for red)
    """
    # Predefined list of 10 visually distinct colors
    color_list = [
        '#FF0000',  # Red
        '#00FF00',  # Green
        '#0000FF',  # Blue
        '#FFFF00',  # Yellow
        '#FF00FF',  # Magenta
        '#00FFFF',  # Cyan
        '#FFA500',  # Orange
        '#800080',  # Purple
        '#008000',  # Dark Green
        '#000080',  # Navy
    ]

    # Return the requested number of colors
    if n <= len(color_list):
        return color_list[:n]
    else:
        # If more colors are requested than in our predefined list,
        # we'll generate additional random colors
        import random
        additional_colors = []
        for _ in range(n - len(color_list)):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            additional_colors.append(f'#{r:02X}{g:02X}{b:02X}')

        return color_list + additional_colors


def run_tobii_calibration(tobii_controller, num_points=5):
    """
    Runs Tobii calibration with the specified number of points.

    Parameters:
        controller (tobii_controller): The initialized Tobii controller.
        num_points (str): Number of calibration points ('5' or '9').

    Returns:
        str: 'success' if calibration completes, 'abort' if aborted.
    """
    # Show Tobii status display
    tobii_controller.show_status()

    # Define calibration points
    x, y = 0.6, 0.4

    num_points = int(num_points)  # Convert to integer
    if num_points == 5:
        calibration_points = [(-x, y), (x, y), (0, 0), (-x, -y), (x, -y)]
    elif num_points == 9:
        calibration_points = [
            (-x, y), (0, y), (x, y),
            (-x, 0), (0, 0), (x, 0),
            (-x, -y), (0, -y), (x, -y)
        ]
    else:
        raise ValueError(
            "Invalid number of calibration points. Use '5' or '9'.")

    # Run calibration
    ret = tobii_controller.run_calibration(calibration_points)

    # Handle calibration result
    if ret == 'abort':
        print("Calibration aborted.")
        return 'abort'
    else:
        print("Calibration completed successfully.")
        return 'success'


def process_gaze_position(current_gaze_position, last_gaze_position):
    """
    Processes the gaze position data and updates the gaze position list.

    Parameters:
        current_gaze_position (list): The latest gaze position data: [lx, ly, rx, ry].
        last_gaze_position: The previous gaze position data.

    Returns:
        tuple: ((eye_x_loc, eye_y_loc), NoEyeData)
    """
    num_of_nan_in_gaze = np.count_nonzero(np.isnan(current_gaze_position))
    no_eye_data = False

    if num_of_nan_in_gaze == 0:  # Both eye data is recorded, take average
        current_gaze_position[0] = (
            current_gaze_position[0] + current_gaze_position[2]) / 2
        current_gaze_position[1] = (
            current_gaze_position[1] + current_gaze_position[3]) / 2
        eye_x_loc = current_gaze_position[0]
        eye_y_loc = current_gaze_position[1]

    elif num_of_nan_in_gaze == 4:  # if no eye data is recorded, copy last gaze position
        no_eye_data = True
        eye_x_loc = last_gaze_position[0]
        eye_y_loc = last_gaze_position[1]

    elif num_of_nan_in_gaze == 2:  # One eye data is recorded, take one eye data
        # Check which eye has data and use it
        if not np.isnan(current_gaze_position[0]) and not np.isnan(current_gaze_position[1]):
            # Left eye has data
            eye_x_loc = current_gaze_position[0]
            eye_y_loc = current_gaze_position[1]
        else:
            # Right eye has data
            eye_x_loc = current_gaze_position[2]
            eye_y_loc = current_gaze_position[3]

    current_gaze_position = np.array((eye_x_loc, eye_y_loc))
    return current_gaze_position, no_eye_data


class Design:
    def __init__(self, number_of_trial=10):
        self.number_of_trial = number_of_trial
        self.moving_mode_seq = None

    def gen_design_interleaved(self):
        """
        Interleaves bouncing/organic mode and locking mode trials.
        E.g., bouncing, locking, organic, locking, bouncing, locking, organic, locking...
        """
        moving_mode_seq = []
        for i in range(self.number_of_trial):
            if i % 2 == 0:
                # Even index: bouncing or organic mode
                if i % 4 == 0:
                    moving_mode_seq.append('bouncing')
                else:
                    moving_mode_seq.append('organic')
            else:
                moving_mode_seq.append('locking')
        self.moving_mode_seq = moving_mode_seq
        return self.moving_mode_seq

    def assign_stim_to_moving_mode(self):
        """
        Randomly assigns stimulus index to each moving mode in the sequence.
        For example, if there are 3 unique moving modes ('bouncing', 'organic', 'locking'),
        randomly assign stimulus 0, 1, and 2 to these modes.

        Returns:
            list: A list of indexes corresponding to the moving_mode_seq.
        """
        if self.moving_mode_seq is None:
            raise ValueError(
                "Moving mode sequence must be generated first. Call gen_design_interleaved() before this method.")

        # Get unique moving modes
        unique_modes = list(set(self.moving_mode_seq))
        nMode = len(unique_modes)

        # Randomly assign stimulus indexes to unique moving modes
        stimulus_indexes = np.random.choice(
            range(nMode), size=nMode, replace=False)

        # Create mapping from moving mode to stimulus index
        self.moving_mode_to_stimulus = {
            mode: stimulus_indexes[i] for i, mode in enumerate(unique_modes)}

        # Create the full stimulus sequence by mapping each moving mode to its assigned stimulus
        stimulus_seq = [self.moving_mode_to_stimulus[mode]
                        for mode in self.moving_mode_seq]
        self.stimulus_sequence = stimulus_seq

        return stimulus_seq


class MovingMode(ABC):
    """
    Abstract base class for different moving modes of the target.
    This class defines the interface for updating and resetting the target's position.
    This class type doesn't deal with presenting the target on the screen.
    Only the logic for movement and position management.
    """

    def __init__(self):
        # Initial position of the target as float64
        self.pos = np.array((0.0, 0.0), dtype=np.float64)

    @abstractmethod
    def update(self):
        """
        Abstract method to update the target's position.
        """
        pass

    def reset(self, pos):
        """
        Abstract method to reset the target's parameters except position.
        """
        self.pos = np.array(
            pos, dtype=np.float64)  # Ensure position is float64


class MovingMode_locking(MovingMode):
    """
    Locking mode for the target.
    The target follows an external position (e.g., mouse, gaze) when close enough and locks to it.
    """

    def __init__(self, win, lock_distance=0.1):
        super().__init__()
        self.win = win
        self.lock_distance = lock_distance
        self.locked = False

    def update(self, position=None):
        """
        Updates the target position based on an external position.

        Parameters:
            external_position (tuple): The external position to potentially lock onto (x, y).
                                      If None, the target remains at its current position.
        """
        # If no external position is provided, keep the current position
        if position is None:
            return

        # If not locked, check the distance to lock the target
        if not self.locked:
            distance = np.sqrt((position[0] - self.pos[0])**2 +
                               (position[1] - self.pos[1])**2)
            if distance <= self.lock_distance:
                self.locked = True  # Lock the target to the external position

        # If locked, update the target position to follow the external position
        if self.locked:
            self.pos = position

    def reset(self, pos=None):
        """
        Resets the target's lock state while preserving position if specified.

        Parameters:
            pos (tuple, optional): New position for the target. If None, keeps current position.
        """
        if pos is not None:
            super().reset(pos)
        self.locked = False


class MovingMode_bouncing(MovingMode):
    """
    Bouncing mode for the target.
    The target moves in a straight line and bounces off the screen boundaries.
    """

    def __init__(self, win, speed=0.01):
        super().__init__()
        self.win = win
        self.speed = speed  # Speed of the target
        # Generate random direction with the specified speed magnitude
        random_angle = np.random.uniform(0, 2 * np.pi)
        self.velocity = np.array([
            self.speed * np.cos(random_angle),
            self.speed * np.sin(random_angle)
        ], dtype=np.float64)
        # Horizontal limit based on window aspect ratio
        self.horizontal_limit = 0.5 * win.aspect
        self.vertical_limit = 0.5  # Vertical limit

    def update(self):
        # Update the target's position based on velocity and check for collisions with screen boundaries
        self.pos += self.velocity
        if self.pos[0] <= -self.horizontal_limit or self.pos[0] >= self.horizontal_limit:
            self.velocity[0] = -self.velocity[0]
        if self.pos[1] <= -self.vertical_limit or self.pos[1] >= self.vertical_limit:
            self.velocity[1] = -self.velocity[1]

    def reset(self, pos=(0, 0)):
        """
        Resets the target's position to the given value and unlocks it.
        """
        super().reset(pos)
        # Generate a new random direction with the same speed magnitude
        random_angle = np.random.uniform(0, 2 * np.pi)
        self.velocity = np.array([
            self.speed * np.cos(random_angle),
            self.speed * np.sin(random_angle)
        ], dtype=np.float64)


class MovingMode_organic(MovingMode):
    """
    Organic movement mode for the target.
    The target wanders randomly with a pull toward the center of the screen
    that is proportional to the square of its distance from the center.
    """

    def __init__(self, win, radius=0.01, speed=0.01, noise_intensity=0.5,
                 max_attraction_strength=0.05, initial_pos=(0, 0)):
        super().__init__()
        self.win = win

        # Initialize attributes with numpy types for consistency
        self.radius = np.array(radius)  # Radius of the stimulus
        self.speed = np.array(speed)    # Speed magnitude of the stimulus
        self.noise_intensity = np.array(
            noise_intensity)  # Random movement intensity
        self.max_attraction_strength = np.array(
            max_attraction_strength)  # Center pull strength

        # Set initial position with numpy arrays
        self.pos = np.array(initial_pos)
        self.x, self.y = self.pos[0], self.pos[1]  # Unpack position

        # Initialize random angle
        self.angle = np.random.uniform(0, 2 * np.pi)  # Random initial angle

        # Calculate boundaries for 'height' units
        self.horizontal_limit = np.array(
            0.5 * win.aspect)  # Horizontal limit
        # Vertical limit in 'height' units
        self.vertical_limit = np.array(0.5)

    def update(self):
        """
        Updates the dot's position and angle based on its movement dynamics.

        - Applies random wander (noise).
        - Adds a pull towards the center proportional to the square of the distance.
        - Handles boundary collisions by reflecting the dot's movement.
        """
        center_x, center_y = 0.0, 0.0  # Center of the screen

        # Calculate the vector from the dot to the center
        dx = self.x - center_x
        dy = self.y - center_y
        dist_from_center = np.sqrt(dx**2 + dy**2)  # Distance from the center

        # Calculate normalized distance (0 at center, 1 at nearest edge)
        max_dist_ref = min(self.horizontal_limit, self.vertical_limit)
        normalized_dist = min(1.0, dist_from_center / max_dist_ref)

        # Calculate attraction strength proportional to the square of the normalized distance
        current_attraction = self.max_attraction_strength * normalized_dist**2

        # Calculate the angle pointing towards the center
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

    def reset(self, pos=(0, 0)):
        """
        Reset the angle
        """
        super().reset(pos)
        # Randomize the angle
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.x, self.y = pos[0], pos[1]  # Unpack position
        self.pos = np.array([self.x, self.y])


class Target(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def set_pos(self, pos):
        pass

    @abstractmethod
    def draw(self):
        pass


class Target_circle(Target):
    def __init__(self, win, radius=0.01, color='yellow'):
        super().__init__()
        self.win = win
        self.radius = radius
        self.color = color
        self.circle = visual.Circle(
            win, radius=self.radius, fillColor=self.color, lineColor=self.color)

    def set_pos(self, pos):
        self.circle.setPos(pos)

    def draw(self):
        self.circle.draw()


class Target_image(Target):
    def __init__(self, win, images=None, scale=0.1):
        super().__init__()
        if images is None:
            self.images = get_images()
        else:
            self.images = images
        self.win = win
        self.scale = scale
        self.image = visual.ImageStim(
            win, image=self.images[0] if self.images else None, size=scale)

    def set_image(self, image_idx):
        """
        Set the image for the target from the images list based on the index.
        """
        if 0 <= image_idx < len(self.images):
            self.image.setImage(self.images[image_idx])
        else:
            raise IndexError("Image index out of range.")

    def set_pos(self, pos):
        """
        Set the position of the image target.
        """
        self.image.setPos(pos)

    def draw(self):
        self.image.draw()


class Controller:
    def __init__(self, controller_type='mouse', win=None):
        self.controller_type = controller_type
        if controller_type == 'tobii':
            self.tobii_controller = psychopy_tobii_controller.tobii_controller(
                win=win)
        elif controller_type == 'mouse':
            self.mouse = event.Mouse()
        else:
            raise ValueError(
                "Invalid controller type. Use 'mouse' or 'tobii'.")

        self.last_pos = np.array([0, 0])
        self.isNoData = False

    def get_pos(self):
        if self.controller_type == 'tobii':
            current_pos = self.tobii_controller.get_current_gaze_position()
            current_pos = np.array(current_pos, dtype=np.float64)
            current_pos, no_eye_data = process_gaze_position(
                current_pos, self.last_pos)
            self.isNoData = no_eye_data

        elif self.controller_type == 'mouse':
            current_pos = self.mouse.getPos()

        self.last_pos = current_pos
        return current_pos

    def record_event(self, event):
        """
        Record an event to the data file.
        """
        if self.controller_type == 'tobii':
            self.tobii_controller.record_event(event)
        elif self.controller_type == 'mouse':
            pass
        print(f"Event recorded: {event}")


class DataManager:
    def __init__(self, data_folder):
        self.data_path_exp = None
        self.data_path_tobii = None
        self.data_path_config = None
        self.data_folder = data_folder
        self.date = datetime.datetime.today().strftime("%Y%m%d%H%M")
        self.iDataEntry = 0  # Initialize iEntry to 0
        self.file = None

    def enter_subj_id(self):
        """
        input subjectID from CLI
        """
        subjectID = input("Enter subject ID: ")
        if not subjectID:
            raise ValueError("Subject ID cannot be empty.")
        self.data_path_exp = os.path.join(
            self.data_folder, f"{subjectID}_exp_{self.date}.csv")
        self.data_path_tobii = os.path.join(
            self.data_folder, f"{subjectID}_tobii_{self.date}.tsv")
        self.data_path_config = os.path.join(
            self.data_folder, f"{subjectID}_config_{self.date}.json")
        print(f"Data paths set: {self.data_path_exp}, {self.data_path_tobii}, {self.data_path_config}")
        
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


def run_exp(controller_type='tobii', target_type='image'):
    # generate design parameters
    design = Design(number_of_trial=config["design"]["number_of_trial"])
    moving_mode_seq = design.gen_design_interleaved()
    stimulus_seq = design.assign_stim_to_moving_mode()

    # ------------------------------ Initialisation ------------------------------ #
    current_pos = np.array([0, 0])  # Initial position of the target

    # Initialize data manager and input subject ID
    data_manager = DataManager(data_folder)
    data_manager.enter_subj_id()
    data_manager.save_config(config)

    # Create a psychopy window
    win = visual.Window(units='height',
                        monitor='default',
                        fullscr=config["experiment"]["fullscreen"], colorSpace='rgb255',
                        color=(100, 100, 100))

    # Hide the mouse if in fullscreen mode
    if config["experiment"]["fullscreen"]:
        win.mouseVisible = False
        event.Mouse(visible=False)

    # initialise target
    if target_type == 'circle':
        target = Target_circle(win)
    elif target_type == 'image':
        target = Target_image(win)
    else:
        raise ValueError("Invalid target type. Use 'circle' or 'image'.")

    # initialise controller
    controller = Controller(controller_type=controller_type, win=win)

    # Initialize position indicator if enabled
    pos_indicator = None
    if config["experiment"]["show_pos_indicator"]:
        pos_indicator = visual.Circle(
            win,
            radius=0.005,  # Small radius for the indicator
            fillColor='red',
            lineColor='red',
            opacity=0.7,  # Semi-transparent
            autoLog=False  # Disable logging for performance
        )

    # initialise text stimulus for instructions
    instructions = visual.TextStim(
        win,
        text="Press 'escape' to exit",
        pos=(0, 0),
        color='white',
        height=0.05
    )

    # Initialize all moving modes with speed from para
    locking_mode = MovingMode_locking(win)
    bouncing_mode = MovingMode_bouncing(win, speed=config["stimulus"]["speed"])
    organic_mode = MovingMode_organic(win, speed=config["stimulus"]["speed"])

    # -------------------------------- calibration ------------------------------- #
    # Run Tobii calibration if using the eye tracker and calibration is enabled
    if controller_type == 'tobii' and config["tobii"]["calibration"]:
        # Message about calibration
        calib_msg = visual.TextStim(
            win,
            text="Eye tracker calibration will start now.\nFollow the dots with your eyes.",
            pos=(0, 0),
            color='white',
            height=0.05
        )
        calib_msg.draw()
        win.flip()
        core.wait(2)

        # Run calibration with specified number of points
        calib_result = run_tobii_calibration(
            controller.tobii_controller,
            config["tobii"]["calibration_points"]
        )

        # Handle calibration result
        if calib_result == 'abort':
            win.close()
            core.quit()
            return

    # --------------------------- Start the experiment --------------------------- #
    if controller_type == 'tobii':
        # Start recording gaze data
        controller.tobii_controller.open_datafile(
            data_manager.data_path_tobii, embed_events=False)
        controller.tobii_controller.subscribe()

    # Record experiment start event
    controller.record_event("Experiment started")

    # welcome message
    instructions.setText("Welcome to the experiment!\nPress 'space' to start")
    instructions.draw()
    win.flip()
    event.waitKeys(keyList=['space'])

    for iTrial, (mode, stim_idx) in enumerate(zip(moving_mode_seq, stimulus_seq)):
        # Record trial start event
        controller.record_event(f"No.{iTrial} trial started")

        # Set the appropriate image for the target if using images
        if target_type == 'image':
            target.set_image(stim_idx)

        # Reset the appropriate moving mode
        current_mode = None
        if mode == 'locking':
            current_mode = locking_mode
        elif mode == 'bouncing':
            current_mode = bouncing_mode
        elif mode == 'organic':
            current_mode = organic_mode

        current_mode.reset(current_pos)

        # Run for the specified trial duration
        start_time = core.getTime()
        frame_num = 0
        while core.getTime() - start_time < config["design"]["trial_duration"]:
            # Get controller position
            controller_pos = controller.get_pos()

            # Update target position based on the current mode
            if mode == 'locking':
                current_mode.update(controller_pos)
            else:
                current_mode.update()

            current_pos = current_mode.pos
            target.set_pos(current_pos)

            # Draw the target, trial counter, and mode display
            target.draw()

            # Draw position indicator if enabled
            if config["experiment"]["show_pos_indicator"]:
                pos_indicator.setPos(controller_pos)
                pos_indicator.draw()

            # Check for escape key
            keys = event.getKeys()
            if 'escape' in keys:
                controller.record_event("Experiment interrupted by user")
                win.close()
                core.quit()
                return

            # log data
            frame_data = {
                'eye_x': controller_pos[0],
                'eye_y': controller_pos[1],
                'stim_x': current_pos[0],
                'stim_y': current_pos[1],
                'no_eye_data': controller.isNoData,
                'trial_num': iTrial,
                'frame_num': frame_num,
                'time': core.getTime(),
                'time_trial': core.getTime() - start_time,
                'moving_mode': mode,
                'stimulus_index': stim_idx,
            }
            data_manager.log_data(frame_data)

            frame_num += 1
            win.flip()

        # Record trial end event
        controller.record_event(f"No.{iTrial} trial ended")

    # Record experiment completion event
    controller.record_event("Experiment completed")

    # Show completion message
    instructions.setText("All trials completed!\nPress 'escape' to exit")
    instructions.draw()
    win.flip()

    # Wait for escape key to exit
    event.waitKeys(keyList=['escape'])

    # ---------------------------- close all resources --------------------------- #
    # Clean up Tobii resources if using eye tracker
    if controller_type == 'tobii':
        controller.tobii_controller.unsubscribe()
        controller.tobii_controller.close_datafile()

    # Close the data file
    data_manager.close_file()

    win.close()
    core.quit()


if __name__ == "__main__":
    run_exp(
        controller_type=config["controller"]['type'], target_type='image')
