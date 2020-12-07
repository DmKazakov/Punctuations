from typing import List, Tuple
import statistics
from collections import deque
from sklearn.preprocessing import StandardScaler

from model.Model import Model
from utils.Window import Window, TagsDistribution


class TagwiseModel(Model):
    def __init__(self, windows_number: int, factory):
        self.factory = factory
        self.models = {}
        self.windows_number = windows_number
        self.scaler = None
        self.windows = None
        self.prev_window = None

    def predict(self, window: Window) -> TagsDistribution:
        if self.prev_window:
            self.fit([self.prev_window, window])
            self.prev_window = window
        X = self.get_vectors(window)
        if len(X) == 0:
            return {}
        if not self.scaler:
            print("No scaler")
            return {}
        # y = self.model.predict(self.scaler.transform(X))
        # return {tag: max(value, 0) for tag, value in zip(window.tags_distribution, y)}
        return {
            tag: max(self.models[tag].predict(self.scaler.transform([x]))[0], 0) for tag, x in X if tag in self.models
        }

    def fit(self, windows: List[Window]):
        if not self.windows:
            self.windows = deque()
            self.windows.extend(windows[:self.windows_number])
            windows = windows[self.windows_number:]
        X, y = self.to_vecs(windows)
        if len(X) == 0:
            return
        if not self.scaler:
            _X = [x for _, x in X]
            self.scaler = StandardScaler()
            self.scaler.fit(_X)

        for (tag, x), l in zip(X, y):
            if tag not in self.models:
                self.models[tag] = self.factory()
            x = self.scaler.transform([x])
            self.models[tag].partial_fit(x, [l])

    def to_vecs(self, windows: List[Window]) -> (List[Tuple[str, List[float]]], List[float]):
        X = []
        y = []
        current = windows[0]

        for next_windows in windows[1:]:
            X.extend(self.get_vectors(current))
            for tag in current.tags_distribution:
                y.append(next_windows.tags_distribution.get(tag, 0))

            self.windows.popleft()
            self.windows.append(current)
            current = next_windows

        return X, y

    def get_vectors(self, window: Window) -> List[Tuple[str, List[float]]]:
        X = []
        last = self.windows.__getitem__(self.windows_number - 1)

        for tag, count in window.tags_distribution.items():
            prev_stats = []
            for window in self.windows:
                prev_stats.append(window.tags_distribution.get(tag, 0))
            x = [
                count,
                window.recent_tags_distribution.get(tag, 0),
                count - last.tags_distribution.get(tag, 0),
                count / last.tags_distribution.get(tag, 1),
                max(prev_stats),
                min(prev_stats),
                statistics.mean(prev_stats)
            ]
            X.append((tag, x))

        return X