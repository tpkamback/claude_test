"""
dataset.py — Toy MME-style dataset generator

Generates synthetic evaluation samples entirely in memory using PIL.
No internet access or file downloads required.

MME format:
  - image: PIL.Image
  - question: str  ("Is there a <object>? Please answer yes or no.")
  - answer: str    ("yes" or "no")
  - task: str      (one of: existence, color, count, position)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from PIL import Image, ImageDraw


@dataclass
class MMESample:
    image: Image.Image
    question: str
    answer: str
    task: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _solid_image(color: tuple[int, int, int], size: int = 224) -> Image.Image:
    return Image.new("RGB", (size, size), color)


def _draw_circle(
    base: Image.Image,
    color: tuple[int, int, int],
    cx: int,
    cy: int,
    r: int = 30,
) -> Image.Image:
    img = base.copy()
    draw = ImageDraw.Draw(img)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(0, 0, 0))
    return img


def _draw_rect(
    base: Image.Image,
    color: tuple[int, int, int],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> Image.Image:
    img = base.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))
    return img


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------

def _make_existence_samples(n: int = 10) -> List[MMESample]:
    """
    Task: existence
    Question: "Is there a red circle in the image? Please answer yes or no."
    Images either contain a red circle (answer=yes) or are plain grey (answer=no).
    Balanced: n//2 yes, n//2 no.
    """
    samples: List[MMESample] = []
    bg = _solid_image((180, 180, 180))  # neutral grey background

    for i in range(n):
        if i < n // 2:
            # Yes: place red circle at centre
            img = _draw_circle(bg, (220, 50, 50), cx=112, cy=112, r=35)
            answer = "yes"
        else:
            # No: plain grey image
            img = bg.copy()
            answer = "no"

        question = "Is there a red circle in the image? Please answer yes or no."
        samples.append(MMESample(image=img, question=question, answer=answer, task="existence"))

    return samples


def _make_color_samples(n: int = 10) -> List[MMESample]:
    """
    Task: color
    Question: "Is the circle blue? Please answer yes or no."
    Images show a circle that is either blue or red.
    Balanced: n//2 yes (blue), n//2 no (red).
    """
    samples: List[MMESample] = []
    bg = _solid_image((200, 200, 200))

    for i in range(n):
        if i < n // 2:
            img = _draw_circle(bg, (50, 80, 220), cx=112, cy=112, r=40)
            answer = "yes"
        else:
            img = _draw_circle(bg, (220, 50, 50), cx=112, cy=112, r=40)
            answer = "no"

        question = "Is the circle blue? Please answer yes or no."
        samples.append(MMESample(image=img, question=question, answer=answer, task="color"))

    return samples


def _make_count_samples(n: int = 10) -> List[MMESample]:
    """
    Task: count
    Question: "Are there exactly two circles in the image? Please answer yes or no."
    Images contain either 2 circles (yes) or 1 circle (no).
    Balanced: n//2 yes, n//2 no.
    """
    samples: List[MMESample] = []
    bg = _solid_image((210, 210, 210))

    for i in range(n):
        if i < n // 2:
            # Two circles
            img = _draw_circle(bg, (80, 160, 80), cx=70, cy=112, r=30)
            img = _draw_circle(img, (80, 160, 80), cx=154, cy=112, r=30)
            answer = "yes"
        else:
            # One circle
            img = _draw_circle(bg, (80, 160, 80), cx=112, cy=112, r=30)
            answer = "no"

        question = "Are there exactly two circles in the image? Please answer yes or no."
        samples.append(MMESample(image=img, question=question, answer=answer, task="count"))

    return samples


def _make_position_samples(n: int = 10) -> List[MMESample]:
    """
    Task: position
    Question: "Is the circle on the left side of the image? Please answer yes or no."
    Images show a circle placed on the left (yes) or right (no) half.
    Balanced: n//2 yes, n//2 no.
    """
    samples: List[MMESample] = []
    bg = _solid_image((215, 215, 215))

    for i in range(n):
        if i < n // 2:
            # Circle on left half (cx ~= 56)
            img = _draw_circle(bg, (160, 80, 200), cx=56, cy=112, r=30)
            answer = "yes"
        else:
            # Circle on right half (cx ~= 168)
            img = _draw_circle(bg, (160, 80, 200), cx=168, cy=112, r=30)
            answer = "no"

        question = "Is the circle on the left side of the image? Please answer yes or no."
        samples.append(MMESample(image=img, question=question, answer=answer, task="position"))

    return samples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_toy_dataset(seed: int = 42) -> List[MMESample]:
    """
    Build and return the full toy MME dataset.

    Returns list of MMESample dicts with keys:
        image    PIL.Image  (224x224 RGB)
        question str
        answer   str        "yes" or "no"
        task     str        "existence" | "color" | "count" | "position"

    Total: 40 samples  (10 per task, balanced yes/no).
    """
    random.seed(seed)

    dataset: List[MMESample] = []
    dataset.extend(_make_existence_samples(10))
    dataset.extend(_make_color_samples(10))
    dataset.extend(_make_count_samples(10))
    dataset.extend(_make_position_samples(10))

    return dataset


if __name__ == "__main__":
    ds = make_toy_dataset()
    from collections import Counter
    task_counts = Counter(s.task for s in ds)
    answer_counts = Counter(s.answer for s in ds)
    print(f"Total samples : {len(ds)}")
    print(f"By task       : {dict(task_counts)}")
    print(f"By answer     : {dict(answer_counts)}")
    # Show first sample per task
    seen = set()
    for s in ds:
        if s.task not in seen:
            seen.add(s.task)
            print(f"\n[{s.task}]")
            print(f"  Q: {s.question}")
            print(f"  A: {s.answer}")
            print(f"  Image: {s.image.size} {s.image.mode}")
