import hashlib
import os
import urllib
from typing import Callable, Optional, Text, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# VAD functionality has been disabled - pyannote.audio dependency removed
def load_vad_model(device, vad_onset=0.500, vad_offset=0.363, use_auth_token=None, model_fp=None):
    raise NotImplementedError("VAD model loading has been disabled. pyannote.audio dependency was removed.")

# VAD classes disabled - pyannote.audio dependency removed

class Segment:
    """Simple segment class for compatibility"""
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker

def merge_chunks(segments, chunk_size, onset: float = 0.5, offset: Optional[float] = None):
    """Simplified merge_chunks without pyannote dependency"""
    raise NotImplementedError("merge_chunks has been disabled. pyannote.audio dependency was removed.")
