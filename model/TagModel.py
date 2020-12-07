from typing import List
import statistics
from collections import deque
from sklearn.preprocessing import StandardScaler

from model.Model import Model
from utils.Window import Window, TagsDistribution


class TagModel(Model):
    def __init__(self, windows_number: int, model, online=True):
        self.model = model
        self.windows_number = windows_number
        self.online = online
        self.scaler = None
        self.windows = None
        self.prev_window = None

    def predict(self, window: Window) -> TagsDistribution:
        if self.prev_window and self.online:
            self.fit([self.prev_window, window])
        self.prev_window = window

        X = self.get_vectors(window)
        if len(X) == 0:
            return {}
        if not self.scaler:
            print("No scaler")
            return {}
        y = self.model.predict(self.scaler.transform(X))
        return {tag: max(value, 0) for tag, value in zip(window.tags_distribution, y)}

    def fit(self, windows: List[Window]):
        if not self.windows:
            self.windows = deque()
            self.windows.extend(windows[:self.windows_number])
            windows = windows[self.windows_number:]
        X, y = self.to_vecs(windows)
        if len(X) == 0:
            return
        if not self.scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(X)

        if self.online:
            self.model.partial_fit(self.scaler.transform(X), y)
        else:
            self.model.fit(self.scaler.transform(X), y)

    def to_vecs(self, windows: List[Window]) -> (List[List[float]], List[float]):
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

    def get_vectors(self, window: Window) -> List[List[float]]:
        X = []
        last = self.windows.__getitem__(self.windows_number - 1)

        for tag, count in window.tags_distribution.items():
            prev_stats = []
            for window in self.windows:
                prev_stats.append(window.tags_distribution.get(tag, 0))
            current_percentage = count / len(window.posts)
            last_count = last.tags_distribution.get(tag, 0)
            last_percentage = last_count / len(last.posts)
            x = [
                count,
                window.recent_tags_distribution.get(tag, 0),
                count - last_count,
                count / 1 if last_count == 0 else last_count,
                current_percentage - last_percentage,
                current_percentage / 1 if last_percentage == 0 else last_percentage,
                max(prev_stats),
                min(prev_stats),
                statistics.mean(prev_stats)
            ]
            X.append(x)

        return X


"""
1. Количество постов с тегом в текущем окне
2. Количество постов с тегом за последний квант времени (квант сильно меньше размера окна)
3'. Разница между количеством постов с тегом
4'. Во сколько раз изменилось количество постов с тегом
Статистика по количеству постов с тегом на N последних окнах:
8. Максимум
9. Минимум
10. Среднее
"""