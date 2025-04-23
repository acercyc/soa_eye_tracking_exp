import os
import numpy as np
import csv
import time
import datetime
from psychopy import visual, core, event
import psychopy_tobii_controller
import GazeStabilizer


def get_valid_filename(raw_filename):
    """
    Convert a string to a valid filename by removing invalid characters.
    """
    # Replace spaces with underscores and remove other invalid filename characters
    valid_filename = raw_filename.replace(' ', '_')
    valid_chars = '-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    valid_filename = ''.join(c for c in valid_filename if c in valid_chars)
    return valid_filename


def process_gaze_position(current_gaze_position, last_pos):
    """
    Process the gaze position data to handle missing values.
    
    Args:
        current_gaze_position: Tuple of (left_x, left_y, right_x, right_y)
        last_pos: Previously recorded position
        
    Returns:
        Tuple of (processed_position, is_no_data_flag)
    """
    num_of_nan_in_gaze = np.count_nonzero(np.isnan(current_gaze_position))
    no_eye_data = False
    
    if num_of_nan_in_gaze == 0:  # Both eyes detected
        # Average the positions from both eyes
        eye_x_loc = (current_gaze_position[0] + current_gaze_position[2]) / 2
        eye_y_loc = (current_gaze_position[1] + current_gaze_position[3]) / 2
    
    elif num_of_nan_in_gaze == 4:  # No eyes detected
        no_eye_data = True
        eye_x_loc = last_pos[0] if last_pos is not None else np.nan
        eye_y_loc = last_pos[1] if last_pos is not None else np.nan
    
    elif num_of_nan_in_gaze == 2:  # Only one eye detected
        # Find which eye has data
        if not np.isnan(current_gaze_position[0]) and not np.isnan(current_gaze_position[1]):
            # Left eye has data
            eye_x_loc = current_gaze_position[0]
            eye_y_loc = current_gaze_position[1]
        else:
            # Right eye has data
            eye_x_loc = current_gaze_position[2]
            eye_y_loc = current_gaze_position[3]
    
    return np.array([eye_x_loc, eye_y_loc]), no_eye_data


def record_gaze(duration, output_dir="data", filename=None, calibrate=True, num_calibration_points=5, show_gaze=True):
    """
    Record gaze position data for the specified duration.
    
    Args:
        duration: Recording duration in seconds
        output_dir: Directory to save the data file
        filename: Name of the output file (without extension)
        calibrate: Whether to perform calibration before recording
        num_calibration_points: Number of calibration points (5 or 9)
        show_gaze: Whether to display the gaze position during recording
        
    Returns:
        Path to the saved data file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"gaze_recording_{timestamp}"
    else:
        filename = get_valid_filename(filename)
    
    # Full path for the output file
    output_path = os.path.join(output_dir, f"{filename}.csv")
    tobii_path = os.path.join(output_dir, f"{filename}_tobii.tsv")
    
    # Create a window
    win = visual.Window(
        units='height', 
        monitor='default', 
        fullscr=True, 
        color=(0.4, 0.4, 0.4)
    )
    
    # Create a Tobii controller
    controller = psychopy_tobii_controller.tobii_controller(win)
    
    # Create visual elements
    instruction_text = visual.TextStim(
        win, 
        text="Press 'space' to start recording or 'escape' to abort", 
        pos=(0, 0), 
        height=0.05, 
        color='white'
    )
    
    countdown_text = visual.TextStim(
        win, 
        text="", 
        pos=(0, 0), 
        height=0.08, 
        color='white'
    )
    
    status_text = visual.TextStim(
        win, 
        text="", 
        pos=(0, -0.4), 
        height=0.03, 
        color='white'
    )
    
    gaze_marker = None
    if show_gaze:
        gaze_marker = visual.Circle(
            win,
            radius=0.01,
            fillColor='red',
            lineColor=None,
            opacity=0.7,
            autoLog=False
        )
        sample_text = visual.TextStim(
            win,
            text='',
            pos=(-0.4, 0.45),
            height=0.03,
            color='white',
            alignHoriz='left'
        )
    
    # Run calibration if enabled
    if calibrate:
        instruction_text.setText("Eye tracker calibration will start now.\nFollow the dots with your eyes.")
        instruction_text.draw()
        win.flip()
        core.wait(2)
        
        # Set up calibration points
        if num_calibration_points == 5:
            x, y = 0.4, 0.4
            calibration_points = [
                (0, 0), (-x, y), (x, y), (-x, -y), (x, -y)
            ]
        elif num_calibration_points == 9:
            x, y = 0.4, 0.4
            calibration_points = [
                (0, y), (-x, y), (x, y),
                (0, 0), (-x, 0), (x, 0),
                (0, -y), (-x, -y), (x, -y)
            ]
        else:
            raise ValueError("Number of calibration points must be 5 or 9")
        
        # Run calibration
        ret = controller.run_calibration(calibration_points)
        if ret == 'abort':
            win.close()
            core.quit()
            return None
    
    # Wait for user to start recording
    instruction_text.setText("Press 'space' to start recording or 'escape' to abort")
    
    waiting = True
    while waiting:
        instruction_text.draw()
        win.flip()
        
        keys = event.getKeys()
        if 'space' in keys:
            waiting = False
        elif 'escape' in keys:
            win.close()
            core.quit()
            return None
    
    # Countdown before recording starts
    # for i in range(3, 0, -1):
    #     countdown_text.setText(str(i))
    #     countdown_text.draw()
    #     win.flip()
    #     core.wait(1)
    
    # Prepare CSV file
    
    stabilizer = GazeStabilizer.MovingAverageStabilizer()
    sample_count = 0
    
    with open(output_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['timestamp', 'time_elapsed', 'gaze_x', 'gaze_y', 
                             'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y', 
                             'no_data_flag'])
    
        # Open tobii data file for raw data
        controller.open_datafile(tobii_path, embed_events=False)
        
        # Start recording
        controller.subscribe()
        controller.record_event("Recording started")
        
        # Record for specified duration
        start_time = time.time()
        last_position = np.array([0.0, 0.0])
        sample_count = 0
        
        while time.time() - start_time < duration:
            sample_count += 1
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            
            # Get gaze data
            current_gaze = controller.get_current_gaze_position()
            
            # Process gaze data
            processed_gaze, no_data = process_gaze_position(current_gaze, last_position)
            last_position = processed_gaze
            
            # Write data to CSV
            timestamp = time.time()
            csv_writer.writerow([
                timestamp, 
                elapsed, 
                processed_gaze[0], 
                processed_gaze[1],
                current_gaze[0],
                current_gaze[1],
                current_gaze[2],
                current_gaze[3],
                no_data
            ])
            csvfile.flush()  # Ensure data is written immediately
            
            # Update display
            if show_gaze and not no_data and not np.isnan(processed_gaze[0]):
                
                # Stabilize gaze position
                stabilized_gaze = stabilizer.stabilize(processed_gaze[0], processed_gaze[1])
                gaze_marker.setPos(stabilized_gaze)
                gaze_marker.draw()
                
                
            # draw sample count
            sample_text.setText(f"Samples: {sample_count}")
            sample_text.draw()
            
            status_text.setText(f"Recording: {elapsed:.1f}s / {duration:.1f}s")
            status_text.draw()
            
            # Check for escape key
            keys = event.getKeys()
            if 'escape' in keys:
                controller.record_event("Recording interrupted by user")
                break
            
            win.flip()
    
        # Stop recording
        controller.record_event("Recording completed")
        controller.unsubscribe()
        controller.close_datafile()
    
    # Show completion message
    instruction_text.setText(f"Recording complete!\nData saved to:\n{output_path}\n\nPress any key to exit")
    instruction_text.draw()
    win.flip()
    
    # Wait for key press to exit
    event.waitKeys()
    
    # Clean up
    win.close()
    core.quit()
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Record eye gaze data using Tobii eye tracker")
    parser.add_argument("-t", "--time", type=float, default=100.0, help="Recording duration in seconds")
    parser.add_argument("-o", "--output", type=str, default="data", help="Output directory")
    parser.add_argument("-f", "--filename", type=str, help="Output filename (without extension)")
    parser.add_argument("--no-calibration", action="store_true", help="Skip calibration")
    parser.add_argument("-p", "--points", type=int, default=5, choices=[5, 9], help="Number of calibration points (5 or 9)")
    parser.add_argument("--no-gaze-display", action="store_true", help="Don't display gaze position during recording")
    
    args = parser.parse_args()
    
    output_path = record_gaze(
        duration=args.time,
        output_dir=args.output,
        filename=args.filename,
        calibrate=args.no_calibration,
        num_calibration_points=args.points,
        show_gaze=not args.no_gaze_display
    )
    
    if output_path:
        print(f"Data saved to: {output_path}")
    else:
        print("Recording was aborted.")
