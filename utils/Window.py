import datetime
from math import sqrt
from typing import Dict, Set, List
import numpy as np

from utils import Post

TagsDistribution = Dict[str, int]


class Window:
    def __init__(self, start: datetime, seconds: int, quantum: int = None):
        self.seconds = seconds
        self.quantum = quantum
        self.start = start
        self.end = start + datetime.timedelta(0, seconds)
        if not quantum:
            self.quantum = max(seconds // 100, 1)
        self.quantum_start = self.end - datetime.timedelta(0, self.quantum)
        self.posts = []
        self.tags_distribution: TagsDistribution = dict()
        self.recent_tags_distribution: TagsDistribution = dict()
        self.distribution_vec = None

    def add_post(self, post: Post) -> bool:
        is_in_bounds = self.start <= post.date < self.end
        if is_in_bounds:
            self.posts.append(post)
            for tag in post.tags:
                self.tags_distribution[tag] = self.tags_distribution.get(tag, 0) + 1

        is_in_quantum_bounds = self.quantum_start <= post.date < self.end
        if is_in_quantum_bounds:
            for tag in post.tags:
                self.recent_tags_distribution[tag] = self.recent_tags_distribution.get(tag, 0) + 1

        return is_in_bounds

    def get_distribution_vec(self, tags: List[str]) -> np.ndarray:
        all_tags = set(tags + list(self.tags_distribution.keys()))
        result = [self.tags_distribution.get(key, 0) for key in sorted(all_tags)]
        return np.array(result)


class Scorer:
    def __init__(self):
        self.previous_tags: Set[str] = set()

    def update(self, tags: List[str]):
        self.previous_tags.update(tags)

    def reset(self):
        self.previous_tags.clear()

    def score(self, predicted: TagsDistribution, actual: TagsDistribution) -> float:
        if len(self.previous_tags) == 0:
            return 0.0
        return sqrt(sum(
            map(lambda tag: pow(predicted.get(tag, 0.0) - actual.get(tag, 0.0), 2), self.previous_tags)
        ) / len(self.previous_tags))


class AccumulativeScorer:
    def __init__(self):
        self.previous_tags: Set[str] = set()
        self.error = 0
        self.samples = 0

    def update(self, tags: List[str]):
        self.previous_tags.update(tags)

    def reset(self):
        self.previous_tags.clear()
        self.error = 0
        self.samples = 0

    def accumulate(self, predicted: TagsDistribution, actual: TagsDistribution):
        if len(self.previous_tags) == 0:
            return
        self.error = sum(map(lambda tag: pow(predicted.get(tag, 0.0) - actual.get(tag, 0.0), 2), self.previous_tags))
        self.samples += len(self.previous_tags)

    def score(self):
        return sqrt(self.error / self.samples)