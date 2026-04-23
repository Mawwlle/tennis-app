"""Feature extraction for EventNet.

Features per frame (14 total):
  0  cx              — normalized ball x ∈ [0, 1]
  1  cy              — normalized ball y ∈ [0, 1]
  2  dx              — velocity x (Δcx)
  3  dy              — velocity y (Δcy)
  4  d²x             — acceleration x
  5  d²y             — acceleration y
  6  speed           — √(dx² + dy²)
  7  angle           — atan2(dy, dx) / π  ∈ [-1, 1]
  8  dist_x_net      — cx − net_cx  (signed: negative = left of net)
  9  dist_y_net      — cy − net_top_y  (signed: negative = above net top)
 10  dist_x_left     — cx − left_player_cx  (distance to left player racket)
 11  dist_y_left     — cy − left_player_cy
 12  dist_x_right    — cx − right_player_cx (distance to right player racket)
 13  dist_y_right    — cy − right_player_cy
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from eventnet.player_detection import PlayerGeometry

N_FEATURES = 14
WINDOW_SIZE = 15


class NetGeometry(BaseModel):
    """Normalized net position averaged over the first N frames of segmentation.

    Defaults model a typical table-tennis camera angle where the net sits
    horizontally centred at ~45 % of frame height.
    """

    cx: float = 0.5     # horizontal centre of net  (normalized)
    top_y: float = 0.45  # top edge of net           (normalized)


def extract_features(
    positions: tuple[tuple[float, float] | None, ...],
    net_geometry: NetGeometry,
    player_geometry: PlayerGeometry | None = None,
) -> NDArray[np.float32]:
    """Convert a window of normalized (cx, cy) positions to kinematic features.

    Args:
        positions:      Window of (cx, cy) | None values, normalized to [0, 1].
        net_geometry:   Averaged net position from segmentation.
        player_geometry: Estimated player/racket positions. Uses defaults when None.

    Returns:
        (N, 14) float32 array.
    """
    players = player_geometry if player_geometry is not None else PlayerGeometry()
    n = len(positions)
    feats = np.zeros((n, N_FEATURES), dtype=np.float32)

    for i, pos in enumerate(positions):
        if pos is not None:
            feats[i, 0], feats[i, 1] = pos

    # dx, dy
    for i in range(1, n):
        if positions[i] is not None and positions[i - 1] is not None:
            feats[i, 2] = feats[i, 0] - feats[i - 1, 0]
            feats[i, 3] = feats[i, 1] - feats[i - 1, 1]

    # d²x, d²y
    for i in range(2, n):
        if positions[i] is not None and positions[i - 1] is not None and positions[i - 2] is not None:
            feats[i, 4] = feats[i, 2] - feats[i - 1, 2]
            feats[i, 5] = feats[i, 3] - feats[i - 1, 3]

    dx, dy = feats[:, 2], feats[:, 3]
    feats[:, 6] = np.sqrt(dx ** 2 + dy ** 2)   # speed
    feats[:, 7] = np.arctan2(dy, dx) / np.pi    # angle ∈ [-1, 1]

    # distance to net and players — only for visible frames
    for i, pos in enumerate(positions):
        if pos is not None:
            feats[i, 8]  = feats[i, 0] - net_geometry.cx        # signed x to net
            feats[i, 9]  = feats[i, 1] - net_geometry.top_y     # signed y to net top
            feats[i, 10] = feats[i, 0] - players.left_cx        # x to left player
            feats[i, 11] = feats[i, 1] - players.left_cy        # y to left player
            feats[i, 12] = feats[i, 0] - players.right_cx       # x to right player
            feats[i, 13] = feats[i, 1] - players.right_cy       # y to right player

    return feats  # (N, 14)
