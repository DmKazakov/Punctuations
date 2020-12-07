from dataclasses import dataclass
import statistics
from typing import List

from model import Model
from utils.Window import Window, Scorer, AccumulativeScorer


@dataclass
class Metrics:
    med_rmse: float
    avg_rmse: float


class MetricsCalculator:
    def __init__(self, windows: List[Window]):
        self.windows = windows
        self.scorer = Scorer()

    def metrics(self, model: Model, log=False) -> Metrics:
        self.scorer.reset()

        rmse = []
        ind = 1
        total = len(self.windows)
        # Итерируемся по парам последовательных окон
        for first, second in zip(self.windows, self.windows[1:]):
            if log:
                print(ind, total)
            ind += 1
            # Обновляем scorer тегами из первого окна, чтобы те учитывались при подсчете ошибки
            self.scorer.update([tag for post in first.posts for tag in post.tags])
            # Заполняем список RMSE для пар окон
            # scorer считает RMSE на всех встретившихся ранее тегах, после чего мы делим его на размер окна,
            # чтобы результаты для разных размеров были сравнимы
            rmse.append(self.scorer.score(model.predict(first), second.tags_distribution) / first.seconds)

        # У полученных результатов считаем медиану и среднее
        return Metrics(statistics.median(rmse), statistics.mean(rmse))
