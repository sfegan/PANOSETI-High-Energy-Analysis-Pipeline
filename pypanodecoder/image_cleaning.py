#!/usr/bin/env python3

# image_cleaning.py: Camera image cleaning algorithms for PANOSETI data.

# Author: Stephen Fegan <sfegan@llr.in2p3.fr> (2026-11-10)
# Laboratoire Leprince-Ringuet, CNRS/IN2P3, Ecole Polytechnique, Institut Polytechnique de Paris

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from functools import lru_cache
from abc import ABC, abstractmethod

from .eventbuilder import CameraImages

__all__ = [
    'get_grid_adjacency',
    'BaseCleaner',
    'PictureBoundaryCleaner',
    'IslandCleaner',
    'CleaningPipeline',
    'clean_picture_boundary',
    'clean_islands',
    'clean_images',
    'invert_cleaned_images'
]


@lru_cache(maxsize=16)
def get_grid_adjacency(shape=(32, 32), connectivity=8):
    """
    Constructs and caches a sparse adjacency matrix for a 2D rectangular pixel grid.

    Args:
        shape (tuple): (rows, cols) dimensions of the camera grid. Default (32, 32).
        connectivity (int): 4 (orthogonal neighbors) or 8 (orthogonal + diagonal neighbors).

    Returns:
        scipy.sparse.csr_matrix: (N_pix, N_pix) boolean adjacency matrix.
    """
    rows, cols = shape
    n_pix = rows * cols
    r, c = np.indices((rows, cols))
    r = r.ravel()
    c = c.ravel()

    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
    else:
        raise ValueError(f"Connectivity must be 4 or 8, got {connectivity}")

    src_list = []
    dst_list = []
    idx = r * cols + c

    for dr, dc in deltas:
        nr = r + dr
        nc = c + dc
        valid = (nr >= 0) & (nr < rows) & (nc >= 0) & (nc < cols)
        src_list.append(idx[valid])
        dst_list.append(nr[valid] * cols + nc[valid])

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    data = np.ones(len(src), dtype=bool)

    return sp.csr_matrix((data, (src, dst)), shape=(n_pix, n_pix), dtype=bool)


def _resolve_threshold(threshold, gti_indexes, n_pix, n_events):
    """
    Resolves thresholds into a 2D array of shape (n_pix, n_events) or (n_pix, 1).

    Args:
        threshold: Scalar, 2D array (rows, cols), 1D array (n_pix,), or dict {gti_index: value}.
        gti_indexes (np.ndarray or None): Event GTI indices of length n_events.
        n_pix (int): Number of pixels per event (e.g. 1024).
        n_events (int): Number of events.

    Returns:
        np.ndarray: Array of shape (n_pix, n_events) or (n_pix, 1) or scalar float.
    """
    if threshold is None:
        return None

    if isinstance(threshold, dict):
        if gti_indexes is None:
            raise ValueError("GTI indexes are required when threshold is provided as a dict.")
        T = np.empty((n_pix, n_events), dtype=float)
        unique_gtis = np.unique(gti_indexes)
        for gti in unique_gtis:
            if gti not in threshold:
                raise KeyError(f"GTI index {gti} not found in threshold dict: {list(threshold.keys())}")
            val = np.asarray(threshold[gti], dtype=float)
            mask = (gti_indexes == gti)
            if val.ndim == 0:
                T[:, mask] = float(val)
            elif val.shape == (32, 32) or val.size == n_pix:
                T[:, mask] = val.reshape(n_pix, 1)
            else:
                raise ValueError(f"Invalid threshold shape for GTI {gti}: {val.shape}")
        return T

    val = np.asarray(threshold, dtype=float)
    if val.ndim == 0:
        return float(val)
    elif val.shape == (32, 32) or val.size == n_pix:
        return val.reshape(n_pix, 1)
    elif val.shape == (n_pix, n_events):
        return val
    else:
        raise ValueError(f"Unsupported threshold shape: {val.shape} for {n_pix} pixels")


def _wrap_result(original_obj, cleaned_images):
    """Wraps cleaned image array back into CameraImages if input was CameraImages."""
    if isinstance(original_obj, CameraImages):
        return CameraImages(
            images=cleaned_images,
            event_times=original_obj.event_times,
            pcap_times=original_obj.pcap_times,
            gti_indexes=original_obj.gti_indexes,
            gti_pcap_times=original_obj.gti_pcap_times,
            quabo_masks=original_obj.quabo_masks,
            gtis=original_obj.gtis,
            events=original_obj.events,
            filter=dict(original_obj.filter),
            source=dict(original_obj.source),
            quabo_pcap_time=original_obj.quabo_pcap_time,
            quabo_event_time=original_obj.quabo_event_time
        )
    return cleaned_images


def invert_cleaned_images(original_images, cleaned_images):
    """
    Inverts cleaning by keeping only the channels that did NOT pass cleaning.

    Channels that passed cleaning (non-NaN in cleaned_images) are set to np.nan.
    Channels that failed cleaning (NaN in cleaned_images, but valid in original_images)
    retain their original charge values.
    Channels that were already NaN in original_images remain NaN.

    Args:
        original_images (CameraImages or np.ndarray): Original images before cleaning.
        cleaned_images (CameraImages or np.ndarray): Cleaned images where rejected pixels are np.nan.

    Returns:
        CameraImages or np.ndarray: New container or array containing only the rejected charges.
    """
    if isinstance(original_images, CameraImages):
        orig_arr = original_images.images
        clean_arr = cleaned_images.images if isinstance(cleaned_images, CameraImages) else cleaned_images
    else:
        orig_arr = np.asarray(original_images)
        clean_arr = cleaned_images.images if isinstance(cleaned_images, CameraImages) else np.asarray(cleaned_images)

    failed_mask = ~np.isnan(orig_arr) & np.isnan(clean_arr)
    inverted_arr = np.where(failed_mask, orig_arr, np.nan)

    return _wrap_result(original_images, inverted_arr)


class BaseCleaner(ABC):
    """Abstract base class for camera image cleaning algorithms."""

    @abstractmethod
    def clean(self, camera_images, **kwargs):
        """
        Cleans image events in camera_images.

        Args:
            camera_images (CameraImages or np.ndarray): Input images container or array.

        Returns:
            CameraImages or np.ndarray: Cleaned image container or array.
        """
        pass

    def __call__(self, camera_images, **kwargs):
        return self.clean(camera_images, **kwargs)


class PictureBoundaryCleaner(BaseCleaner):
    """
    Two-level Picture / Boundary cleaning algorithm.

    Pixels with charge >= picture_threshold are marked as picture (core) pixels.
    Pixels with charge >= boundary_threshold that are adjacent to at least one
    picture pixel are marked as boundary pixels.
    Isolated picture pixels are retained by default.
    All other channels below threshold are set to np.nan.

    If invert=True, inverts the selection: channels that do NOT pass cleaning are
    preserved, while channels passing cleaning are set to np.nan.

    Pixel adjacency checking across all events is fully vectorized via SciPy sparse
    graph matrix multiplication for maximum performance without Python loops.
    """
    def __init__(self, picture_threshold, boundary_threshold, connectivity=8,
                 min_picture_neighbors=0, invert=False):
        """
        Args:
            picture_threshold (float, np.ndarray, or dict): Picture threshold level(s).
                Can be a scalar, a 2D (32, 32) array, or a dict {gti_index: threshold}.
            boundary_threshold (float, np.ndarray, or dict): Boundary threshold level(s).
                Can be a scalar, a 2D (32, 32) array, or a dict {gti_index: threshold}.
            connectivity (int): 8 (default) or 4 pixel connectivity.
            min_picture_neighbors (int): Minimum number of picture neighbors required
                for a picture pixel to survive. Default 0 (isolated picture pixels retained).
            invert (bool): If True, retains only charges that do NOT pass cleaning. Default False.
        """
        self.picture_threshold = picture_threshold
        self.boundary_threshold = boundary_threshold
        self.connectivity = connectivity
        self.min_picture_neighbors = min_picture_neighbors
        self.invert = invert

    def clean(self, camera_images, **kwargs):
        """
        Cleans the images in camera_images according to Picture/Boundary criteria.

        Args:
            camera_images (CameraImages or np.ndarray): Input container or array.
            invert (bool, optional): Overrides self.invert if specified.

        Returns:
            CameraImages or np.ndarray: Cleaned container or array with rejected pixels set to np.nan.
        """
        do_invert = kwargs.get('invert', self.invert)

        if isinstance(camera_images, CameraImages):
            images = camera_images.images
            gti_indexes = camera_images.gti_indexes
        else:
            images = np.asarray(camera_images)
            gti_indexes = kwargs.get('gti_indexes', None)

        is_2d = (images.ndim == 2)
        if is_2d:
            images = images[:, :, np.newaxis]
            if gti_indexes is not None:
                gti_indexes = np.atleast_1d(gti_indexes)

        rows, cols, n_events = images.shape
        if n_events == 0:
            return _wrap_result(camera_images, images if not is_2d else images[:, :, 0])

        n_pix = rows * cols
        Q = images.reshape(n_pix, n_events)

        t_pic = _resolve_threshold(self.picture_threshold, gti_indexes, n_pix, n_events)
        t_bnd = _resolve_threshold(self.boundary_threshold, gti_indexes, n_pix, n_events)

        # Ignore NaNs during comparisons
        p_pic = ~np.isnan(Q) & (Q >= t_pic)
        p_bnd = ~np.isnan(Q) & (Q >= t_bnd)

        # Vectorized neighbor lookup via SciPy sparse graph adjacency matrix
        adj = get_grid_adjacency((rows, cols), self.connectivity)

        # (adj @ p_pic) computes number of picture neighbors for every pixel in every event
        n_pic_neighbors = adj @ p_pic

        # Picture pixels condition
        if self.min_picture_neighbors > 0:
            surviving_pic = p_pic & (n_pic_neighbors >= self.min_picture_neighbors)
        else:
            surviving_pic = p_pic

        # Boundary pixels condition: must border at least one picture pixel
        surviving_bnd = p_bnd & (n_pic_neighbors > 0)

        keep = surviving_pic | surviving_bnd

        if do_invert:
            # Preserve only channels that do NOT pass cleaning and were not already NaN
            selected = ~keep & ~np.isnan(Q)
        else:
            selected = keep & ~np.isnan(Q)

        cleaned_Q = np.where(selected, Q, np.nan)
        cleaned_images = cleaned_Q.reshape(rows, cols, n_events)

        if is_2d:
            cleaned_images = cleaned_images[:, :, 0]

        return _wrap_result(camera_images, cleaned_images)


class IslandCleaner(BaseCleaner):
    """
    Island (connected component) cleaning algorithm.

    Segments active channels into connected components ("islands") using
    SciPy sparse graph connected component analysis.
    Islands containing >= min_channels channels are preserved.
    Islands smaller than min_channels are rejected and set to np.nan.

    If invert=True, inverts the selection: preserves channels from islands smaller
    than min_channels, setting surviving islands to np.nan.
    """
    def __init__(self, min_channels=2, threshold=None, connectivity=8,
                 only_largest_island=False, chunk_size=5000, invert=False):
        """
        Args:
            min_channels (int): Minimum number of channels required in an island to survive.
                Default is 2 (rejects isolated single-channel islands).
            threshold (float, np.ndarray, dict, optional): Optional threshold applied to
                define active channels. If None, all non-NaN channels are treated as active.
            connectivity (int): 8 (default) or 4 pixel connectivity.
            only_largest_island (bool): If True, retains only the largest island in each event.
                Default False (retains all islands meeting min_channels).
            chunk_size (int): Event chunk size for batched graph connected component processing.
            invert (bool): If True, retains only charges that do NOT pass cleaning. Default False.
        """
        self.min_channels = min_channels
        self.threshold = threshold
        self.connectivity = connectivity
        self.only_largest_island = only_largest_island
        self.chunk_size = chunk_size
        self.invert = invert

    def clean(self, camera_images, **kwargs):
        """
        Cleans the images in camera_images according to Island criteria.

        Args:
            camera_images (CameraImages or np.ndarray): Input container or array.
            invert (bool, optional): Overrides self.invert if specified.

        Returns:
            CameraImages or np.ndarray: Cleaned container or array with rejected pixels set to np.nan.
        """
        do_invert = kwargs.get('invert', self.invert)

        if isinstance(camera_images, CameraImages):
            images = camera_images.images
            gti_indexes = camera_images.gti_indexes
        else:
            images = np.asarray(camera_images)
            gti_indexes = kwargs.get('gti_indexes', None)

        is_2d = (images.ndim == 2)
        if is_2d:
            images = images[:, :, np.newaxis]
            if gti_indexes is not None:
                gti_indexes = np.atleast_1d(gti_indexes)

        rows, cols, n_events = images.shape
        if n_events == 0:
            return _wrap_result(camera_images, images if not is_2d else images[:, :, 0])

        n_pix = rows * cols
        Q = images.reshape(n_pix, n_events)

        if self.threshold is not None:
            t_val = _resolve_threshold(self.threshold, gti_indexes, n_pix, n_events)
            active_mask = ~np.isnan(Q) & (Q >= t_val)
        else:
            active_mask = ~np.isnan(Q)

        # Process in chunks to ensure bounded memory and maximum vectorized speed
        surviving_mask = np.zeros_like(active_mask)
        adj = get_grid_adjacency((rows, cols), self.connectivity)
        adj_coo = adj.tocoo()
        src_p = adj_coo.row
        dst_p = adj_coo.col

        for start in range(0, n_events, self.chunk_size):
            end = min(start + self.chunk_size, n_events)
            sub_mask = active_mask[:, start:end]
            sub_N = end - start

            if not np.any(sub_mask):
                continue

            pix_idx, ev_idx = np.nonzero(sub_mask)
            active_per_event = np.bincount(ev_idx, minlength=sub_N)
            event_valid = (active_per_event >= self.min_channels)

            if not np.any(event_valid):
                continue

            valid_pair_mask = event_valid[ev_idx]
            sub_pix_idx = pix_idx[valid_pair_mask]
            sub_ev_idx = ev_idx[valid_pair_mask]

            n_nodes = len(sub_pix_idx)
            if n_nodes == 0:
                continue

            node_ids = np.full((n_pix, sub_N), -1, dtype=np.int32)
            node_ids[sub_pix_idx, sub_ev_idx] = np.arange(n_nodes, dtype=np.int32)

            src_nodes = node_ids[src_p, :]
            dst_nodes = node_ids[dst_p, :]
            edge_mask = (src_nodes >= 0) & (dst_nodes >= 0)
            u = src_nodes[edge_mask]
            v = dst_nodes[edge_mask]

            big_adj = sp.csr_matrix((np.ones(len(u), dtype=bool), (u, v)), shape=(n_nodes, n_nodes), dtype=bool)
            n_comp, labels = csgraph.connected_components(big_adj, directed=False)
            counts = np.bincount(labels)

            if self.only_largest_island:
                for ev in np.unique(sub_ev_idx):
                    ev_node_mask = (sub_ev_idx == ev)
                    ev_labels = labels[ev_node_mask]
                    unique_ev_labels = np.unique(ev_labels)
                    ev_counts = counts[unique_ev_labels]
                    best_idx = np.argmax(ev_counts)
                    best_label = unique_ev_labels[best_idx]
                    if ev_counts[best_idx] >= self.min_channels:
                        keep_nodes = ev_node_mask & (labels == best_label)
                        surviving_mask[sub_pix_idx[keep_nodes], start + ev] = True
            else:
                valid_labels = np.flatnonzero(counts >= self.min_channels)
                if len(valid_labels) > 0:
                    surviving_nodes = np.isin(labels, valid_labels)
                    surviving_mask[sub_pix_idx[surviving_nodes], start + sub_ev_idx[surviving_nodes]] = True

        if do_invert:
            selected = active_mask & ~surviving_mask
        else:
            selected = surviving_mask

        cleaned_Q = np.where(selected, Q, np.nan)
        cleaned_images = cleaned_Q.reshape(rows, cols, n_events)

        if is_2d:
            cleaned_images = cleaned_images[:, :, 0]

        return _wrap_result(camera_images, cleaned_images)


class CleaningPipeline(BaseCleaner):
    """Chains multiple cleaning algorithms sequentially."""

    def __init__(self, *cleaners, invert=False):
        """
        Args:
            *cleaners (BaseCleaner): Sequence of cleaners to apply in order.
            invert (bool): If True, inverts the final pipeline selection, keeping charges
                that failed any stage of the pipeline. Default False.
        """
        self.cleaners = list(cleaners)
        self.invert = invert

    def add(self, cleaner):
        """Appends a cleaner to the pipeline."""
        self.cleaners.append(cleaner)
        return self

    def clean(self, camera_images, **kwargs):
        """Runs camera_images through all cleaners in the pipeline."""
        do_invert = kwargs.get('invert', self.invert)
        res = camera_images
        for cleaner in self.cleaners:
            res = cleaner.clean(res, invert=False)

        if do_invert:
            return invert_cleaned_images(camera_images, res)
        return res


def clean_picture_boundary(camera_images, picture_threshold, boundary_threshold,
                           connectivity=8, min_picture_neighbors=0, invert=False):
    """
    Cleans camera images using two-level Picture / Boundary threshold cleaning.

    Args:
        camera_images (CameraImages or np.ndarray): Input image container or array.
        picture_threshold (float, np.ndarray, or dict): Picture threshold level(s).
        boundary_threshold (float, np.ndarray, or dict): Boundary threshold level(s).
        connectivity (int): 8 (default) or 4 pixel connectivity.
        min_picture_neighbors (int): Minimum picture neighbors required for picture pixels.
            Default 0 (isolated picture pixels survive).
        invert (bool): If True, inverts cleaning to retain only charges that do NOT pass cleaning.

    Returns:
        CameraImages or np.ndarray: Cleaned container or array with rejected pixels set to np.nan.
    """
    cleaner = PictureBoundaryCleaner(
        picture_threshold=picture_threshold,
        boundary_threshold=boundary_threshold,
        connectivity=connectivity,
        min_picture_neighbors=min_picture_neighbors,
        invert=invert
    )
    return cleaner.clean(camera_images)


def clean_islands(camera_images, min_channels=2, threshold=None, connectivity=8,
                  only_largest_island=False, invert=False):
    """
    Cleans camera images by filtering connected components ("islands").

    Args:
        camera_images (CameraImages or np.ndarray): Input image container or array.
        min_channels (int): Minimum number of channels required in an island to survive.
            Default is 2 (filters isolated single-channel noise).
        threshold (float, np.ndarray, dict, optional): Threshold applied to define active
            channels. If None, all non-NaN channels are treated as active.
        connectivity (int): 8 (default) or 4 pixel connectivity.
        only_largest_island (bool): If True, retains only the largest island in each event.
        invert (bool): If True, inverts cleaning to retain only charges that do NOT pass cleaning.

    Returns:
        CameraImages or np.ndarray: Cleaned container or array with rejected pixels set to np.nan.
    """
    cleaner = IslandCleaner(
        min_channels=min_channels,
        threshold=threshold,
        connectivity=connectivity,
        only_largest_island=only_largest_island,
        invert=invert
    )
    return cleaner.clean(camera_images)


def clean_images(camera_images, method='picture_boundary', **kwargs):
    """
    Convenience dispatcher for image cleaning algorithms.

    Args:
        camera_images (CameraImages or np.ndarray): Input image container or array.
        method (str or BaseCleaner): Cleaning algorithm name ('picture_boundary', 'islands',
            'pipeline') or an instance of BaseCleaner.
        **kwargs: Arguments passed to the chosen cleaner (including invert=True/False).

    Returns:
        CameraImages or np.ndarray: Cleaned image container or array.
    """
    if isinstance(method, BaseCleaner):
        return method.clean(camera_images, **kwargs)

    method_lower = str(method).lower().strip()
    if method_lower in ('picture_boundary', 'tailcuts', 'two_level'):
        return clean_picture_boundary(camera_images, **kwargs)
    elif method_lower in ('islands', 'island'):
        return clean_islands(camera_images, **kwargs)
    elif method_lower in ('pipeline', 'both'):
        pic_thresh = kwargs.pop('picture_threshold')
        bnd_thresh = kwargs.pop('boundary_threshold')
        min_chan = kwargs.pop('min_channels', 2)
        conn = kwargs.pop('connectivity', 8)
        invert = kwargs.pop('invert', False)
        pipe = CleaningPipeline(
            PictureBoundaryCleaner(pic_thresh, bnd_thresh, connectivity=conn),
            IslandCleaner(min_channels=min_chan, connectivity=conn, **kwargs),
            invert=invert
        )
        return pipe.clean(camera_images)
    else:
        raise ValueError(
            f"Unknown cleaning method: '{method}'. "
            f"Supported methods are 'picture_boundary', 'islands', 'pipeline'."
        )
