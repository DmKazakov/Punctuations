import abc
from typing import Callable, List
from collections import Counter
import numpy as np

from utils.Window import Window, TagsDistribution, Scorer


class Model(abc.ABC):
    @abc.abstractmethod
    def predict(self, window: Window) -> TagsDistribution:
        pass

    @abc.abstractmethod
    def fit(self, windows: List[Window]):
        pass


class WindowModel(abc.ABC):
    @abc.abstractmethod
    def predict(self, window: Window) -> TagsDistribution:
        pass

    @abc.abstractmethod
    def fit(self, window: Window, distribution: TagsDistribution):
        pass


class Baseline(Model):
    def predict(self, window: Window) -> TagsDistribution:
        return window.tags_distribution

    def fit(self, windows: List[Window]):
        pass


class ZeroModel(Model):
    def predict(self, window: Window) -> TagsDistribution:
        return {}

    def fit(self, windows: List[Window]):
        pass


class Ensemble(Model):
    def __init__(self, models_number: int, train_size: int, factory: Callable[[], Model]):
        self.size = models_number
        self.train_size = train_size
        self.factory = factory
        self.models = []
        self.pending_model = None
        self.prev_windows = []
        self.scorer = Scorer()

    def predict(self, window: Window) -> TagsDistribution:
        self.scorer.update([tag for post in window.posts for tag in post.tags])
        if self.prev_windows is not None:
            self.fit(self.prev_windows[1:] + [window])

        predicted = Counter()
        for model in self.models:
            predicted.update(model.predict(window))
        return {k: round(v / len(self.models)) for k, v in predicted.most_common()}

    def fit(self, windows: List[Window]):
        for i in range(len(windows) - self.train_size + 1):
            model = self.factory()
            model.fit(windows[i: i + self.train_size])

            if len(self.models) < self.size and model:
                self.models.append(model)
                return

            if self.pending_model:
                X = windows[i + self.train_size - 2]
                y = windows[i + self.train_size - 1].tags_distribution
                score = self.scorer.score(self.pending_model.predict(X), y)
                scores = [self.scorer.score(model.predict(X), y) for model in self.models]
                max_score_ind = np.argmax(scores)
                if score < scores[max_score_ind]:
                    scores[max_score_ind] = self.pending_model

            if model:
                self.pending_model = model

        self.prev_windows = windows[-self.train_size:]