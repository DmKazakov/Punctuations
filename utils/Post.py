from __future__ import annotations
import re
from dataclasses import dataclass

from typing import Set, Callable
from dateutil.parser import parse
import datetime

tag_pattern = re.compile("#[^# ,.\n]+")


@dataclass
class Post:
    tags: Set[str]
    date: datetime

    @staticmethod
    def parse(text: str, date: str, to_label: Callable[[str], str] = lambda s: s) -> Post:
        tags = tag_pattern.findall(text.lower())
        tags = set(map(to_label, tags))
        date_time = parse(date, ignoretz=True)
        return Post(tags, date_time)


def to_first_letter(tag: str) -> str:
    return tag[1]


def to_same_tag(tag: str) -> str:
    return tag


def to_numbered_group(groups_number: int):
    group_size = int((ord('z') - ord('a') + 1) / (groups_number - 1))
    if group_size == 0:
        raise Exception("Groups number is too large")

    def do(tag: str) -> str:
        if ord('a') <= ord(tag[1]) <= ord('z'):
            n = ord(tag[1]) - ord('a')
            group_number = n // group_size + 1
            if group_number >= groups_number:
                return str(groups_number - 1)
            return str(group_number)
        else:
            return str(groups_number)
    return do