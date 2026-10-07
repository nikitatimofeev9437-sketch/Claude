"""Hard content guard: the program only depicts adults.

Prompts (and LoRA file names) that reference minors are rejected outright, and
age-related terms are always appended to the negative prompt. This is not a
setting and cannot be switched off from the UI.
"""
from __future__ import annotations

import re

_EN = [
    r"child(ren)?", r"kids?", r"toddlers?", r"infants?", r"bab(y|ies)", r"underage", r"under[- ]?age",
    r"minors?", r"pre[- ]?teens?", r"teens?", r"teenagers?", r"tweens?", r"juvenile", r"adolescents?",
    r"loli(con|ta)?", r"shota(con)?", r"cub", r"young (girl|boy)s?", r"little (girl|boy)s?",
    r"(school ?girl|school ?boy)s?", r"elementary", r"middle ?school", r"junior high", r"kindergarten",
    r"preschool", r"(girl|boy)s? aged? \d+", r"aged? ?(1[0-7]|[1-9])\b", r"(1[0-7]|[1-9]) ?(yo|y/o|years? old)",
    r"petite child", r"jailbait", r"nymphet",
]
# Russian stems, matched from the start of a word.
_RU = [
    r"реб[её]н", r"дет(и|ей|ск)", r"малыш", r"младен", r"девоч", r"мальчик", r"школьн", r"подрост",
    r"несовершеннолет", r"малолет", r"лол[иь]", r"шота", r"тинейдж", r"юн(ая|ый|ой|ую|ые)",
]

_PATTERN = re.compile(
    r"\b(?:" + "|".join(_EN) + r")\b|\b(?:" + "|".join(_RU) + r")",
    re.IGNORECASE,
)

ALWAYS_NEGATIVE = "child, kid, teen, underage, minor, loli, young-looking, childlike"


class BlockedPrompt(ValueError):
    pass


def find_blocked(text: str) -> list[str]:
    # Underscores join booru tags and file names ("loli_style") — treat them as spaces.
    text = (text or "").replace("_", " ")
    return sorted({m.group(0).lower() for m in _PATTERN.finditer(text)})


def check(*texts: str) -> None:
    hits: list[str] = []
    for t in texts:
        hits += find_blocked(t)
    if hits:
        raise BlockedPrompt(
            "Запрос отклонён: найдены слова, связанные с несовершеннолетними ("
            + ", ".join(sorted(set(hits)))
            + "). Программа генерирует только взрослых персонажей."
        )


def harden_negative(negative: str) -> str:
    negative = (negative or "").strip().rstrip(",")
    return f"{negative}, {ALWAYS_NEGATIVE}" if negative else ALWAYS_NEGATIVE
