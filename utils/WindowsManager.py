from typing import List

from utils.Post import Post
from utils.Window import Window


class WindowsManager:
    def __init__(self, posts: List[Post], quantum=None):
        self.posts = posts
        self.min_time = min([p.date for p in posts])
        self.max_time = max([p.date for p in posts])
        self.quantum = quantum

    def windows_sizes_range(self, min_windows_size: int, range_size: int, min_windows_number: int) -> range:
        seconds_total = (self.max_time - self.min_time).total_seconds()
        max_window_size = seconds_total // min_windows_number
        step_size = (max_window_size - min_windows_size) // (range_size - 1)
        return range(min_windows_size, int(max_window_size), int(step_size))

    def windows(self, window_size: int):
        windows = []
        current = self.min_time
        while current <= self.max_time:
            window = Window(current, window_size, self.quantum)
            windows.append(window)
            current = window.end

        for post in self.posts:
            window_index = (post.date - self.min_time).total_seconds() / window_size
            if not windows[int(window_index)].add_post(post):
                raise Exception("Invalid windows list")

        windows = [w for w in windows if len(w.posts) > 0 and any(len(post.tags) > 0 for post in w.posts)]
        if windows[-1].end > self.min_time:
            return windows[:-1]
        else:
            return windows
