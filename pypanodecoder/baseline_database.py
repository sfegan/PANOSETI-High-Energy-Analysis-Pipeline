#!/usr/bin/env python3

# baseline_database.py: Quabo pedestal baseline measurement database.

# Author: Stephen Fegan <sfegan@llr.in2p3.fr> (2026-06-09)
# Laboratoire Leprince-Ringuet, CNRS/IN2P3, Ecole Polytechnique, Institut Polytechnique de Paris

import os
import json
import glob
import bisect
import numpy as np

from .pcapdecoder import parse_time
from .eventbuilder import CameraEvent

__all__ = ['BaselineDatabase']


class BaselineDatabase:
    """
    A database of Quabo pedestal baseline measurements.
    Loads measurements from quabo_ph_baseline.json files and maps UIDs to
    board locations using quabo_uids.json files.
    """
    def __init__(self, paths=None):
        """
        Args:
            paths (str, list, optional): Glob pattern(s) or list of files/directories
                                         to search for baseline files.
        """
        # board_loc -> [ (timestamp, baseline_array), ... ] sorted by timestamp
        self.baselines = {}
        if paths:
            self.load(paths)

    def load(self, paths):
        """
        Searches for and loads baseline measurements.

        Args:
            paths (str, list): Glob pattern(s) or list of files/directories.
        """
        if isinstance(paths, str):
            if any(c in paths for c in '*?['):
                files = glob.glob(paths, recursive=True)
            else:
                files = [paths]
        else:
            files = []
            for p in paths:
                if any(c in p for c in '*?['):
                    files.extend(glob.glob(p, recursive=True))
                else:
                    files.append(p)

        # Collect all quabo_ph_baseline.json files
        baseline_files = []
        for f in files:
            if os.path.isdir(f):
                baseline_files.extend(glob.glob(os.path.join(f, "**/quabo_ph_baseline.json"), recursive=True))
            elif f.endswith("quabo_ph_baseline.json"):
                baseline_files.append(f)

        for bfile in sorted(baseline_files):
            # Look for quabo_uids.json in the same directory
            bdir = os.path.dirname(bfile)
            ufile = os.path.join(bdir, "quabo_uids.json")

            if not os.path.exists(ufile):
                # Try parent directory as a fallback
                ufile = os.path.join(os.path.dirname(bdir), "quabo_uids.json")
                if not os.path.exists(ufile):
                    continue

            uid_map = self._load_uid_map(ufile)
            self._load_baseline_file(bfile, uid_map)

    def _load_uid_map(self, ufile):
        """Loads quabo_uids.json and returns a map of UID -> board_loc."""
        with open(ufile, 'r') as f:
            data = json.load(f)

        uid_to_loc = {}
        for dome in data.get('domes', []):
            for module in dome.get('modules', []):
                ip = module.get('ip_addr', '')
                try:
                    octets = [int(o) for o in ip.split('.')]
                    if len(octets) == 4:
                        # board_loc logic from pcapdecoder.py
                        board_loc_base = ((octets[2] & 0x03) << 8) | octets[3]
                        # Each module has 4 quabos, usually sequentially indexed                                                                                                                                           [352/1091]
                        for i, q in enumerate(module.get('quabos', [])):
                            uid = q.get('uid')
                            if uid:
                                uid_to_loc[uid] = board_loc_base + i
                except (ValueError, IndexError):
                    continue
        return uid_to_loc

    def _load_baseline_file(self, bfile, uid_map):
        """Loads a baseline file and adds its data to the database."""
        with open(bfile, 'r') as f:
            data = json.load(f)

        try:
            ts = parse_time(data.get('date'))
        except (ValueError, TypeError):
            # Use file modification time if date is missing or invalid
            ts = os.path.getmtime(bfile)

        for q in data.get('quabos', []):
            uid = q.get('uid')
            coefs = q.get('coefs')
            if uid in uid_map and coefs:
                loc = uid_map[uid]
                if loc not in self.baselines:
                    self.baselines[loc] = []

                # Check if this timestamp already exists for this board
                exists = False
                for existing_ts, _ in self.baselines[loc]:
                    if abs(existing_ts - ts) < 1e-3:
                        exists = True
                        break

                if not exists:
                    # Store as numpy array for easier manipulation
                    self.baselines[loc].append((ts, np.array(coefs)))
                    self.baselines[loc].sort(key=lambda x: x[0])

    def get_baseline(self, time, board_loc=None, telescope_id=None):
        """
        Retrieves the nearest baseline measurement before the given time.

        Args:
            time (float, str): UTC time (timestamp or ISO string).
            board_loc (int, optional): Specific board location.
            telescope_id (int, optional): Retrieve combined baseline image for a telescope.

        Returns:
            np.ndarray: 256-pixel array (if board_loc given) or
                        32x32 camera image (if telescope_id given).
                        Returns None if no baseline found.
        """
        ts_req = parse_time(time)

        if board_loc is not None:
            return self._get_board_baseline(ts_req, board_loc)

        if telescope_id is not None:
            # Reconstruct 32x32 image from the 4 quabos (0-3) of this telescope
            q_baselines = []
            for i in range(4):
                loc = (telescope_id << 2) | i
                q_baselines.append(self._get_board_baseline(ts_req, loc))

            # Check if we have at least one quabo
            if all(b is None for b in q_baselines):
                return None

            return CameraEvent.make_image(
                quabo0_pix=q_baselines[0],
                quabo1_pix=q_baselines[1],
                quabo2_pix=q_baselines[2],
                quabo3_pix=q_baselines[3]
            )

        return None

    def _get_board_baseline(self, ts, loc):
        if loc not in self.baselines or not self.baselines[loc]:
            return None

        # Find the index of the first baseline with timestamp > ts
        times = [b[0] for b in self.baselines[loc]]
        idx = bisect.bisect_right(times, ts)

        # We want the one BEFORE or AT ts, which is idx - 1
        if idx == 0:
            # No baseline before this time, return the earliest one anyway?
            # User says "return the nearest measured baseline that is before any given time"
            # If nothing is before, maybe we shouldn't return anything.
            return None

        return self.baselines[loc][idx - 1][1]

    def get_adjacent_times(self, time, board_loc=None, telescope_id=None):
        """
        Returns the times of the previous and next measurements.

        Returns:
            tuple: (prev_time, next_time). None if not available.
        """
        ts_req = parse_time(time)
        times = self.get_all_times(board_loc=board_loc, telescope_id=telescope_id)
        if not times:
            return (None, None)

        idx = bisect.bisect_left(times, ts_req)

        if idx < len(times) and abs(times[idx] - ts_req) < 1e-4:
            prev_idx = idx - 1
            next_idx = idx + 1
        else:
            prev_idx = idx - 1
            next_idx = idx

        prev_time = times[prev_idx] if prev_idx >= 0 else None
        next_time = times[next_idx] if next_idx < len(times) else None

        return (prev_time, next_time)

    def get_all_times(self, board_loc=None, telescope_id=None):
        """
        Returns a sorted list of all unique baseline measurement times.

        Args:
            board_loc (int, optional): Filter for a specific board.
            telescope_id (int, optional): Filter for a specific telescope (any of its 4 quabos).

        Returns:
            list: Sorted list of Unix timestamps (floats).
        """
        times = set()
        if board_loc is not None:
            if board_loc in self.baselines:
                times.update(b[0] for b in self.baselines[board_loc])
        elif telescope_id is not None:
            for i in range(4):
                loc = (telescope_id << 2) | i
                if loc in self.baselines:
                    times.update(b[0] for b in self.baselines[loc])
        else:
            for bl_list in self.baselines.values():
                times.update(b[0] for b in bl_list)

        return sorted(list(times))
