"""Shared fixtures."""

from __future__ import annotations

import math
from typing import List

import pytest

from closed_loop.track import StartPose
from tests.synthetic import RING_CENTER_M, make_ring
from track_utils import Cone


@pytest.fixture
def ring() -> List[Cone]:
    return make_ring()


@pytest.fixture
def ring_start() -> StartPose:
    return StartPose(RING_CENTER_M, 0.0, math.pi / 2.0)
