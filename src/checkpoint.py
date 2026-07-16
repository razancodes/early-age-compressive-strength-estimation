"""
Crash-resilient training checkpoint system.

Provides a TrainingCheckpoint class that persists pipeline progress after each
major work unit (subset × model combination). On resume, completed units are
skipped and the pipeline picks up where it crashed.

Key design decisions:
  - Atomic writes: checkpoint is written to a .tmp file, then os.replace()'d
    to prevent corruption if a crash occurs mid-write.
  - Work-unit granularity: one checkpoint per (subset, model_type) pair. This
    caps worst-case re-work to ~1 hour (a single nested-CV run).
  - Argument validation: on resume, warns if CLI args (e.g. --n-trials) differ
    from the original run to prevent silent inconsistencies.
"""

import copy
import json
import os
import pickle
import time
from datetime import datetime
from typing import Any, Optional


class TrainingCheckpoint:
    """
    Manages pipeline checkpoint state for crash-resilient training.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to store checkpoint files.
    pipeline_name : str
        Identifier for the pipeline (e.g. 'stage_a', 'stage_b').
    """

    def __init__(self, checkpoint_dir: str = 'checkpoints',
                 pipeline_name: str = 'pipeline'):
        self.checkpoint_dir = checkpoint_dir
        self.pipeline_name = pipeline_name
        self._filepath = os.path.join(
            checkpoint_dir, f'{pipeline_name}_checkpoint.pkl'
        )
        self._tmp_filepath = self._filepath + '.tmp'

        # State
        self.completed_units: set = set()
        self.all_results: list = []
        self.all_best_params: dict = {}
        self.pipeline_args: dict = {}
        self.timestamp: str = ''
        self.extra: dict = {}  # For stage-specific data (stacking, GP, etc.)

    # ── Core API ────────────────────────────────────────────────────────

    def save(self):
        """
        Atomically save checkpoint state to disk.

        Writes to a temporary file first, then replaces the target file.
        This ensures the checkpoint is never in a half-written state.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.timestamp = datetime.now().isoformat()

        state = {
            'completed_units': self.completed_units,
            'all_results': self.all_results,
            'all_best_params': self.all_best_params,
            'pipeline_args': self.pipeline_args,
            'timestamp': self.timestamp,
            'extra': self.extra,
        }

        # Write to temp file first
        with open(self._tmp_filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Atomic rename (os.replace is atomic on both POSIX and Windows NTFS)
        os.replace(self._tmp_filepath, self._filepath)

    def load(self) -> bool:
        """
        Load checkpoint state from disk.

        Returns
        -------
        bool
            True if a checkpoint was loaded, False if no checkpoint exists.
        """
        if not os.path.exists(self._filepath):
            return False

        try:
            with open(self._filepath, 'rb') as f:
                state = pickle.load(f)

            self.completed_units = state.get('completed_units', set())
            self.all_results = state.get('all_results', [])
            self.all_best_params = state.get('all_best_params', {})
            self.pipeline_args = state.get('pipeline_args', {})
            self.timestamp = state.get('timestamp', '')
            self.extra = state.get('extra', {})
            return True

        except (pickle.UnpicklingError, EOFError, KeyError) as e:
            print(f"  WARNING: Checkpoint file is corrupted ({e}). "
                  f"Starting fresh.")
            return False

    def clean(self):
        """Remove checkpoint files and reset state."""
        for path in (self._filepath, self._tmp_filepath):
            if os.path.exists(path):
                os.remove(path)
        self.completed_units = set()
        self.all_results = []
        self.all_best_params = {}
        self.pipeline_args = {}
        self.timestamp = ''
        self.extra = {}
        print(f"  Checkpoint cleared: {self._filepath}")

    # ── Work-unit tracking ──────────────────────────────────────────────

    @staticmethod
    def make_key(*parts: str) -> str:
        """Build a checkpoint key from parts, e.g. ('XGBoost', 'Full')."""
        return '_'.join(parts)

    def is_complete(self, key: str) -> bool:
        """Check if a work unit has been completed."""
        return key in self.completed_units

    def mark_complete(self, key: str):
        """Mark a work unit as completed and persist immediately."""
        self.completed_units.add(key)
        self.save()

    def add_results(self, results_rows: list, best_params: Optional[dict] = None,
                    params_key: Optional[str] = None):
        """
        Append result rows and optionally store best hyperparameters.

        Parameters
        ----------
        results_rows : list[dict]
            One or more metric dictionaries to append.
        best_params : dict, optional
            Best hyperparameters to store.
        params_key : str, optional
            Key for best_params (e.g. 'XGBoost_Full').
        """
        self.all_results.extend(results_rows)
        if best_params is not None and params_key is not None:
            self.all_best_params[params_key] = best_params

    # ── Argument validation ─────────────────────────────────────────────

    def validate_args(self, current_args: dict):
        """
        Compare current CLI args with the checkpoint's stored args.
        Warns on mismatches that could cause inconsistent results.
        """
        if not self.pipeline_args:
            return  # No stored args to compare

        critical_keys = ['n_trials', 'data', 'quick']
        for key in critical_keys:
            stored = self.pipeline_args.get(key)
            current = current_args.get(key)
            if stored is not None and current is not None and stored != current:
                print(f"  WARNING: Argument '--{key.replace('_', '-')}' changed: "
                      f"{stored} → {current}")
                print(f"           Completed units used the old value. "
                      f"Consider --clean for consistency.")

    # ── Status printing ─────────────────────────────────────────────────

    def print_status(self):
        """Print human-readable checkpoint status."""
        n_complete = len(self.completed_units)
        if n_complete == 0:
            print("  No completed work units found in checkpoint.")
            return

        print(f"  Checkpoint loaded: {self._filepath}")
        print(f"  Last saved:        {self.timestamp}")
        print(f"  Completed units:   {n_complete}")
        for unit in sorted(self.completed_units):
            print(f"    [OK] {unit}")
        print(f"  Accumulated results: {len(self.all_results)} rows")
        print(f"  Stored params:       {len(self.all_best_params)} entries")
        print()

    @property
    def exists(self) -> bool:
        """Check if a checkpoint file exists on disk."""
        return os.path.exists(self._filepath)
