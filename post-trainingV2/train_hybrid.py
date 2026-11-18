import os
import gzip
import random
import argparse
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import chess
from datasets import load_dataset
from tqdm import tqdm

from chess_hw import load_model, save_model, ChessTokenizer
from chess_hw.engine import choose_move as engine_choose_move

