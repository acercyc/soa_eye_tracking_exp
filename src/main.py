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
        "stabilizer_type": "moving_average",  # Stabilizer type. none, 'moving_average'
        "stabilizer_moving_average": {
            "buffer_size": 50,  # Size of the moving average buffer
            "sampling_rate": 500,  # Sampling rate in Hz
        },

    },
    "stimulus": {
        "target_type": "image",  # Options: 'circle' or 'image'
        "speed": 0.005,  # Speed of the target in bouncing mode
        "scale": 0.05,  # Scale of the target image
        "flash": True,  # Whether to flash the target at the beginning of a trial
        "sound": {
            "play": True,  # Whether to play sound
        },
        "images": [
            "src/images/inosisi.png",
            "src/images/kani.png",
            "src/images/kirin.png",
            "src/images/mouse.png",
            "src/images/panda.png",
            "src/images/tatsu.png",
        ],
        "sounds": [
            "src/sounds/nada-9-326002.wav",
            "src/sounds/notify-1-310752.wav",
            "src/sounds/notify-169186.wav",
            "src/sounds/zapsplat_bell_small_handbell_service_bell_ring_medium_sequence_112480.wav",
        ],
    },
    "runtime":{}
}

# initialise sound devices
if config["stimulus"]["sound"]["play"]:
    from psychopy import prefs
    # prefs.hardware['audioLib'] = ['sounddevice']
    # prefs.hardware['audioLatencyMode'] = '1' 
    prefs.hardware['audioLib'] = ['pygame']  
    prefs.hardware['audioSampleRate'] = 44100
    from psychopy import sound


# create data directory if not exist
def create_data_directory():
    """
    Ensures that the data directory exists. If it doesn't, creates it.
    """
    data_folder = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../data'))
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Data directory created: {data_folder}")
    else:
        print(f"Data directory already exists: {data_folder}")
    return data_folder


# Get all image file paths in the /image directory
def get_images():
    return config["stimulus"]["images"]

def get_sounds():
    return config["stimulus"]["sounds"]


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

def flash_target(target, win, flashes=5, on_duration=0.2, off_duration=0.1):
    """
    Flash the target to capture attention.
    
    Parameters:
        target: The target object to flash
        win: The window to display on
        flashes: Number of times to flash (default: 3)
        on_duration: Duration in seconds to show the target (default: 0.2)
        off_duration: Duration in seconds to hide the target (default: 0.1)
    """
    for i in range(flashes):
        # Show target
        target.draw()
        win.flip()
        core.wait(on_duration)
        
        # Hide target (just flip the window without drawing the target)
        win.flip()
        core.wait(off_duration)
        
def bring_back_to_screen(pos, horizontal_limit, vertical_limit):
    """
    Bring the target back to the screen if it goes out of bounds.
    """
    # Check if the target is out of bounds and bring it back
    if pos[0] < -horizontal_limit:
        pos[0] = -horizontal_limit
    elif pos[0] > horizontal_limit:
        pos[0] = horizontal_limit

    if pos[1] < -vertical_limit:
        pos[1] = -vertical_limit
    elif pos[1] > vertical_limit:
        pos[1] = vertical_limit

    return pos

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
    
    def assign_moving_mode_to_image(self):
        """randomly assign unique image file to moving mode"""
        if self.moving_mode_seq is None:
            raise ValueError(
                "Moving mode sequence must be generated first. Call gen_design_interleaved() before this method.")

        # Get unique moving modes
        unique_modes = list(set(self.moving_mode_seq))
        nMode = len(unique_modes)
        
        # Randomly assign image indexes to unique moving modes
        image_indexes = np.random.choice(
            range(len(config["stimulus"]["images"])), size=nMode, replace=False)
        
        # Create mapping from moving mode to image file
        self.moving_mode_to_image = {
            mode: config["stimulus"]["images"][image_indexes[i]] for i, mode in enumerate(unique_modes)}
        
        # save this mapping to config runtime  
        config["runtime"]["moving_mode_to_image"] = self.moving_mode_to_image
        return self.moving_mode_to_image
        


    def assign_moving_mode_to_sound(self):
        """randomly assign unique sound file to moving mode"""
        if self.moving_mode_seq is None:
            raise ValueError(
                "Moving mode sequence must be generated first. Call gen_design_interleaved() before this method.")

        # Get unique moving modes
        unique_modes = list(set(self.moving_mode_seq))
        nMode = len(unique_modes)

        # Randomly assign sound indexes to unique moving modes
        sound_indexes = np.random.choice(
            range(len(config["stimulus"]["sounds"])), size=nMode, replace=False)

        # Create mapping from moving mode to sound file
        self.moving_mode_to_sound = {
            mode: config["stimulus"]["sounds"][sound_indexes[i]] for i, mode in enumerate(unique_modes)}

        # save this mapping to config runtime
        config["runtime"]["moving_mode_to_sound"] = self.moving_mode_to_sound
        return self.moving_mode_to_sound


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
        # Add boundary limits like in other moving modes
        self.horizontal_limit = 0.5 * win.aspect - config["experiment"]["screen_margin"]
        self.vertical_limit = 0.5 - config["experiment"]["screen_margin"]

    def update(self, position=None, margin_control=False):
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
            # Get position but constrain it within screen boundaries
            x = position[0]
            y = position[1]
            
            if margin_control:
                # Constrain x within horizontal limits
                x = max(-self.horizontal_limit, min(x, self.horizontal_limit))
                
                # Constrain y within vertical limits
                y = max(-self.vertical_limit, min(y, self.vertical_limit))
            
            # Update position with constrained values
            self.pos = np.array([x, y], dtype=np.float64)

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
    def set_stim(self, stim):
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
        
    def set_stim(self, stim_idx):
        """
        Set the stimulus for the target. In this case, it's a circle, so we can set its color.
        """
        if stim_idx < len(get_colors()):
            self.color = get_colors()[stim_idx]
            self.circle.setFillColor(self.color)
            self.circle.setLineColor(self.color)
        else:
            raise IndexError("Stimulus index out of range.")

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
        
    def set_stim(self, image_path):
        """
        Set the image for the target. 
        image_path: str, path to the image file
        """
        self.image.setImage(image_path)
        self.image.setSize(self.scale)


    def set_pos(self, pos):
        """
        Set the position of the image target.
        """
        self.image.setPos(pos)

    def draw(self):
        self.image.draw()


class Sound_effect:
    """
    A class to handle sound effects.
    """
    def __init__(self, moving_mode_to_sound):

        
        self.moving_mode_to_sound = moving_mode_to_sound
        self.sounds = {}
        self.preload_sound()
        self.isPlayed = False
    
    def preload_sound(self):
        """
        Preload sound files for each moving mode.
        """
        self.sounds = {}
        for mode, sound_path in self.moving_mode_to_sound.items():
            if os.path.exists(sound_path):
                self.sounds[mode] = sound.Sound(sound_path)
            else:
                print(f"Sound file not found: {sound_path}")
                
    def play(self, mode):
        """
        Play the sound associated with the given moving mode.
        """
        if mode in self.sounds:
            if not self.isPlayed:
                self.sounds[mode].play()
                self.isPlayed = True
        else:
            print(f"No sound available for mode: {mode}")
            
    def reset(self):
        """
        Reset the sound effect state.
        """
        self.isPlayed = False
        # stop all sounds
        for sound in self.sounds.values():
            sound.stop()
    

class ControllerBase(ABC):
    @abstractmethod
    def get_pos(self):
        pass

    @abstractmethod
    def record_event(self, event):
        pass


class TobiiController(ControllerBase):
    def __init__(self, win, stabilizer_type=None):
        self.tobii_controller = psychopy_tobii_controller.tobii_controller(
            win=win)
        self.last_pos = np.array([0, 0], dtype=np.float64)
        self.isNoData = False
        self.stabilizer_type = stabilizer_type
        self.win = win

        if stabilizer_type == 'moving_average':
            self._bg_on = threading.Event()
            self.bg_sampling_rate = config["tobii"]["stabilizer_moving_average"]["sampling_rate"]
            self.bg_buffer_size = config["tobii"]["stabilizer_moving_average"]["buffer_size"]
            self.stabilizer = GazeStabilizer.MovingAverageStabilizer(
                self.bg_buffer_size)
            self._bg_thread = threading.Thread(
                target=self._run_background_sampling, daemon=True)
        else:
            self.stabilizer = None
            self._bg_thread = None
            self.bg_sampling_rate = None
            self.bg_buffer_size = None

    def subscribe(self, data_path_tobii=None):
        self.tobii_controller.open_datafile(
            data_path_tobii, embed_events=False)
        self.tobii_controller.subscribe()
        if self._bg_thread:
            self._bg_on.set()  # Set the event to start background sampling
            self._bg_thread.start()

    def unsubscribe(self):
        self.tobii_controller.unsubscribe()
        self.tobii_controller.close_datafile()
        if self._bg_thread:
            self._bg_on.clear()
            self._bg_thread.join()

    def _run_background_sampling(self):
        period = 1.0 / self.bg_sampling_rate
        while self._bg_on.is_set():
            # Get the current gaze position
            pos = self.tobii_controller.get_current_gaze_position()
            pos = np.array(pos, dtype=np.float64)
            pos, no_eye_data = process_gaze_position(pos, self.last_pos)
            self.isNoData = no_eye_data
            self.last_pos = pos

            # Apply the stabilizer if available
            if self.stabilizer:
                pos = self.stabilizer.stabilize(pos[0], pos[1])

            # wait for the next sampling period
            time.sleep(period)

    def get_pos(self):
        pos = self.tobii_controller.get_current_gaze_position()
        pos = np.array(pos, dtype=np.float64)
        pos, no_eye_data = process_gaze_position(pos, self.last_pos)
        self.isNoData = no_eye_data
        self.last_pos = pos

        # Apply the stabilizer if available
        if self.stabilizer:
            pos = self.stabilizer.stabilize(pos[0], pos[1])
            pos = np.array(pos, dtype=np.float64)
        return pos

    def record_event(self, event):
        self.tobii_controller.record_event(event)
        print(f"Event recorded: {event}")


class MouseController(ControllerBase):
    def __init__(self, win=None):
        self.mouse = event.Mouse(win=win)

    def get_pos(self):
        return self.mouse.getPos()

    def record_event(self, event):
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
        print(
            f"Data paths set: {self.data_path_exp}, {self.data_path_tobii}, {self.data_path_config}")

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
            
            

def run_exp(controller_type='tobii'):
    # define target type
    target_type = config["stimulus"]["target_type"]    
    
    # create date directory if not exist
    data_folder = create_data_directory()
    
    # generate design parameters
    design = Design(number_of_trial=config["design"]["number_of_trial"])
    moving_mode_seq = design.gen_design_interleaved()
    design.assign_moving_mode_to_image()

    # Create Sound_effect class
    if config["stimulus"]["sound"]["play"]:
        moving_mode_to_sound = design.assign_moving_mode_to_sound()
        sound_effect = Sound_effect(moving_mode_to_sound)


     # Initial position of the target
    current_pos = np.array([0, 0]) 

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
        target = Target_image(win, scale=config["stimulus"]["scale"])
    else:
        raise ValueError("Invalid target type. Use 'circle' or 'image'.")


    # initialise controller
    if controller_type == 'tobii':
        controller = TobiiController(
            win=win, stabilizer_type=config["tobii"]["stabilizer_type"])
    else:
        controller = MouseController(win=win)

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
        controller.subscribe(data_manager.data_path_tobii)

    # Record experiment start event
    controller.record_event("Experiment started")

    # welcome message
    instructions.setText("Welcome to the experiment!\nPress 'space' to start")
    instructions.draw()
    win.flip()
    event.waitKeys(keyList=['space'])
    
    win.flip()
    core.wait(2)

    for iTrial, mode in enumerate(moving_mode_seq):
        # Record trial start event
        controller.record_event(f"No.{iTrial} trial started")

        # Set the appropriate image for the target if using images
        if target_type == 'image':
            # Use the image mapping instead of index-based selection
            image_file = design.moving_mode_to_image[mode]
            target.set_stim(image_file)

        # reset sound_effect
        if config["stimulus"]["sound"]["play"]:
            sound_effect.reset()

        # Reset the appropriate moving mode
        current_mode = None
        if mode == 'locking':
            current_mode = locking_mode
        elif mode == 'bouncing':
            current_mode = bouncing_mode
        elif mode == 'organic':
            current_mode = organic_mode

        # Reset the position within screen limits
        horizontal_limit = 0.5 * win.aspect - config["experiment"]["screen_margin"]
        vertical_limit = 0.5 - config["experiment"]["screen_margin"]
        current_pos = bring_back_to_screen(
            current_pos, horizontal_limit, vertical_limit)
        current_mode.reset(current_pos)
        target.set_pos(current_pos)

        # Play sound if enabled
        if config["stimulus"]["sound"]["play"]:
            sound_effect.play(mode)
        
        # flash the target if enable
        if config["stimulus"]["flash"]:
            flash_target(target, win)

        # reset timer
        start_time = core.getTime()
        frame_num = 0
        effective_time = 0
        current_time = 0
        last_time = start_time
        while effective_time < config["design"]["trial_duration"]:
            # get time
            current_time = core.getTime()
            dt = current_time - last_time
            last_time = current_time
            
            # Get controller position
            controller_pos = controller.get_pos()
            
            # if no eye data is recorded, time is ignored
            if not controller.isNoData:
                effective_time += dt

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

            # log data - update to use mode directly instead of stim_idx
            frame_data = {
                'eye_x': controller_pos[0],
                'eye_y': controller_pos[1],
                'stim_x': current_pos[0],
                'stim_y': current_pos[1],
                'no_eye_data': controller.isNoData if controller_type == 'tobii' else False,
                'trial_num': iTrial,
                'frame_num': frame_num,
                'time': core.getTime(),
                'time_trial': core.getTime() - start_time,
                'moving_mode': mode,
                'image_file': design.moving_mode_to_image[mode] if target_type == 'image' else 'None',
                'sound_file': design.moving_mode_to_sound[mode] if config["stimulus"]["sound"]["play"] else 'None',
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
        controller.unsubscribe()

    # Close the data file
    data_manager.close_file()

    win.close()
    core.quit()


if __name__ == "__main__":
    run_exp(controller_type=config["controller"]['type'])
