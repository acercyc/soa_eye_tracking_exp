"""
Write a class that uses the Tobii controller to sample data from the Tobii eye tracker in parallel using psychopy_tobii_controller
The sampling rate is adjustable and the data is stored in a buffer.
The buffer size is adjustable and the data is stored in a circular buffer.
"""
import time
import threading
from collections import deque
import psychopy_tobii_controller
from psychopy import visual, core, event


class ParallelTobiiSampler:
    def __init__(self, win, sampling_rate=60.0, buffer_size=1000):
        """
        :param win: PsychoPy window instance for tobii_controller
        :param sampling_rate: samples per second
        :param buffer_size: max number of stored samples
        """
        self.controller = psychopy_tobii_controller.tobii_controller(win)
        self.sampling_rate = sampling_rate
        self.buffer = deque(maxlen=buffer_size)  # (timestamp, (lx, ly, rx, ry))
        self._stop_evt = threading.Event()
        self._thread = None

    def start(self):
        """Begin sampling in background."""
        self.controller.subscribe()
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        period = 1.0 / self.sampling_rate
        prev_ts = None
        while not self._stop_evt.is_set():
            ts = time.time()
            gaze = self.controller.get_current_gaze_position()
            # print sampling interval
            if prev_ts is not None:
                print(f"Sampling interval: {ts - prev_ts:.3f}s\tGaze: {gaze}")            
            self.buffer.append((ts, gaze))
            prev_ts = ts
            # time.sleep(period)

    def stop(self):
        """Stop sampling and clean up."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join()
        self.controller.unsubscribe()

    def get_buffer(self):
        """
        Retrieve a snapshot of the current buffer.
        :return: list of (timestamp, gaze_tuple)
        """
        return list(self.buffer)

if __name__ == "__main__":

    # Create a PsychoPy window
    win = visual.Window([800, 600], monitor="testMonitor", units="deg")

    # Initialize the sampler
    sampler = ParallelTobiiSampler(win, sampling_rate=60.0, buffer_size=1000)
    try:
        sampler.start()
        while True:
            if 'escape' in event.getKeys():  # Check for 'escape' key press
                break
            core.wait(0.1)  # Small delay to reduce CPU usage
    finally:
        sampler.stop()
        win.close()