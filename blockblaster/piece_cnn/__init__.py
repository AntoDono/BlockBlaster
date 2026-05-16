"""CNN-based queue piece recognition for the Block Blast assist GUI.

Public surface:

  * :func:`render_piece_sample`, :func:`generate_batch` — synthetic data
    generation for training (no labelled real images required).
  * :class:`PieceCNN` — the small classifier architecture.
  * :class:`PieceClassifier` — convenience wrapper that loads weights and
    classifies a list of BGR slot crops.
"""

from blockblaster.piece_cnn.model import (
    DEFAULT_WEIGHT_PATH,
    PieceCNN,
    PieceClassifier,
    preprocess_batch,
    preprocess_bgr,
)
from blockblaster.piece_cnn.synth import (
    EMPTY_CLASS_ID,
    INPUT_SIZE,
    NUM_CLASSES,
    NUM_PIECES,
    class_id_for,
    generate_batch,
    piece_for_class,
    render_piece_sample,
)

__all__ = [
    "DEFAULT_WEIGHT_PATH",
    "EMPTY_CLASS_ID",
    "INPUT_SIZE",
    "NUM_CLASSES",
    "NUM_PIECES",
    "PieceCNN",
    "PieceClassifier",
    "class_id_for",
    "generate_batch",
    "piece_for_class",
    "preprocess_batch",
    "preprocess_bgr",
    "render_piece_sample",
]
