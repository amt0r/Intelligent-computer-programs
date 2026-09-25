"""Simple stopwatch utility for measuring training time."""

import time


class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0.0
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.start_time = time.perf_counter()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.elapsed_time += time.perf_counter() - self.start_time
            self.is_running = False
        return self.elapsed_time

    def reset(self):
        self.start_time = None
        self.elapsed_time = 0.0
        self.is_running = False

    def get_time(self):
        if self.is_running:
            return self.elapsed_time + (time.perf_counter() - self.start_time)
        return self.elapsed_time
