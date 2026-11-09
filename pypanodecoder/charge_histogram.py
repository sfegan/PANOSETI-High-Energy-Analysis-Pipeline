#!/usr/bin/env python3

# charge_histogram.py: Multi-dimensional histogram of charge values and robust statistics.

# Author: Stephen Fegan <sfegan@llr.in2p3.fr> (2026-05-13)
# Laboratoire Leprince-Ringuet, CNRS/IN2P3, Ecole Polytechnique, Institut Polytechnique de Paris

import numpy as np

__all__ = ['ChargeHistogram']


class ChargeHistogram:
    """
    Represents a multi-dimensional histogram of charge values.
    """
    def __init__(self, qcenter, qhist, bin_width):
        self.qcenter = qcenter
        self.qhist = qhist
        self.bin_width = bin_width

    @classmethod
    def gaussian(cls, sigma, qcenter, mean=None):
        """
        Generates a Gaussian ChargeHistogram.

        Args:
            sigma (array-like): Standard deviation(s) of the Gaussian(s).
            qcenter (array-like): Bin centers for the histogram.
            mean (array-like, optional): Mean(s) of the Gaussian(s).
                                       Defaults to None (all zeros).

        Returns:
            ChargeHistogram: A new histogram containing Gaussian densities.
        """
        from scipy.stats import norm
        sigma = np.asanyarray(sigma)
        if mean is None:
            mean = np.zeros_like(sigma)
        else:
            mean = np.asanyarray(mean)

        if len(qcenter) < 2:
            raise ValueError("qcenter must have at least two values to determine bin_width")

        bin_width = qcenter[1] - qcenter[0]
        edges = np.append(qcenter - 0.5 * bin_width, qcenter[-1] + 0.5 * bin_width)

        # Broadcast mean and sigma to (grid..., 1) and edges to (N+1,)
        # scipy.stats.norm.cdf handles the broadcasting
        cdf_edges = norm.cdf(edges, loc=mean[..., np.newaxis], scale=sigma[..., np.newaxis])

        # Probabilities in each bin
        probabilities = np.diff(cdf_edges, axis=-1)

        # Normalize as density: divide by bin_width
        qhist = probabilities / bin_width

        return cls(qcenter, qhist, bin_width)

    @property
    def shape(self):
        """Returns the shape of the histogram grid (e.g., (32, 32))."""
        return self.qhist.shape[:-1]

    def extract(self, *index):
        """
        Extracts a single element (or sub-grid) from the histogram grid.

        Args:
            index: The index or slice to extract.

        Returns:
            ChargeHistogram: A new histogram containing only the selected elements.
        """
        # qhist is (grid_shape..., num_bins)
        # We slice the grid dimensions; the last dimension (bins) is preserved.
        new_qhist = self.qhist[index]
        return ChargeHistogram(self.qcenter, new_qhist, self.bin_width)

    def downsample(self, factor):
        """
        Downsamples the histogram by combining adjacent bins.

        Args:
            factor (int): The number of bins to combine.

        Returns:
            ChargeHistogram: A new downsampled histogram.
        """
        if factor <= 1:
            return self

        n_bins = self.qhist.shape[-1]
        n_new_bins = n_bins // factor

        if n_new_bins == 0:
            # Return a single bin covering everything if factor > n_bins
            new_qcenter = np.array([self.qcenter.mean()])
            new_qhist = self.qhist.sum(axis=-1, keepdims=True)
            return ChargeHistogram(new_qcenter, new_qhist, self.bin_width * n_bins)

        truncated_n = n_new_bins * factor

        # Reshape and sum the histogram
        new_qhist = self.qhist[..., :truncated_n].reshape(self.shape + (n_new_bins, factor)).sum(axis=-1)

        # Reshape and average the bin centers
        new_qcenter = self.qcenter[:truncated_n].reshape(n_new_bins, factor).mean(axis=-1)

        return ChargeHistogram(new_qcenter, new_qhist, self.bin_width * factor)

    def __add__(self, other):
        """
        Sums two ChargeHistograms together.
        The histograms must have the same grid shape, bin centers, and bin width.
        """
        if not isinstance(other, ChargeHistogram):
            return NotImplemented

        if self.shape != other.shape:
            raise ValueError(f"Incompatible shapes: {self.shape} vs {other.shape}")

        if self.bin_width != other.bin_width:
            raise ValueError(f"Incompatible bin widths: {self.bin_width} vs {other.bin_width}")

        if not np.array_equal(self.qcenter, other.qcenter):
            # Fallback to check if they are "close enough" if needed,
            # but usually they should be identical.
            raise ValueError("Bin centers do not match")

        return ChargeHistogram(self.qcenter, self.qhist + other.qhist, self.bin_width)

    def __mul__(self, other):
        """
        Performs a numerical convolution of two ChargeHistograms across the grid.
        This is mapped to the multiplication operator (*).

        The result represents the distribution of the sum of two independent
        random variables, each distributed according to one of the histograms.

        Note: This assumes the binning is uniform and identical.
        The resulting histogram will be centered around the sum of the means,
        but since we force it back into the same qcenter range,
        values falling outside the range will be lost (truncated).
        """
        if not isinstance(other, ChargeHistogram):
            return NotImplemented

        if self.shape != other.shape:
            raise ValueError(f"Incompatible shapes: {self.shape} vs {other.shape}")

        if self.bin_width != other.bin_width:
            raise ValueError(f"Incompatible bin widths: {self.bin_width} vs {other.bin_width}")

        if not np.array_equal(self.qcenter, other.qcenter):
            raise ValueError("Bin centers do not match")

        # Numerical convolution along the last axis (the histogram bins)
        from scipy.signal import fftconvolve

        # qhist is (grid..., bins)
        new_qhist = fftconvolve(self.qhist, other.qhist, mode='full', axes=-1)

        # Clip small negative values from FFT floating-point artifacts
        new_qhist = np.maximum(new_qhist, 0)

        n_bins = len(self.qcenter)
        q0 = self.qcenter[0]

        # Alignment logic:
        # The first bin of 'full' convolution corresponds to value 2*q0.
        # We want to extract bins starting at value q0.
        # The index of value q0 in the 'full' array is (q0 - 2*q0) / bin_width = -q0 / bin_width.
        # This ensures that a delta function at value 0 (if present) acts as the identity.
        offset = int(round(-q0 / self.bin_width))

        if offset < 0:
            crop_start = 0
            fill_start = -offset
        else:
            crop_start = offset
            fill_start = 0

        final_qhist = np.zeros_like(self.qhist, dtype=float)

        crop_end = min(crop_start + n_bins - fill_start, new_qhist.shape[-1])
        fill_end = fill_start + (crop_end - crop_start)

        if fill_start < n_bins:
            final_qhist[..., fill_start:fill_end] = new_qhist[..., crop_start:crop_end]

        # Normalization to keep the y-scale (counts) roughly consistent.
        # Dividing by the sum of 'other' ensures that if 'other' is a delta-like
        # kernel, the total counts of 'self' are preserved.
        total_other = np.sum(other.qhist, axis=-1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            final_qhist = np.where(total_other > 0, final_qhist / total_other, final_qhist)

        return ChargeHistogram(self.qcenter, final_qhist, self.bin_width)

    def normalize(self, density=True):
        """
        Normalizes the histogram.

        Args:
            density (bool): If True, the integral of the histogram will be 1
                            (divided by sum and bin width).
                            If False, the sum of the bins will be 1.

        Returns:
            ChargeHistogram: A new normalized histogram.
        """
        total = np.sum(self.qhist, axis=-1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            new_qhist = np.where(total > 0, self.qhist / total, self.qhist).astype(float)
            if density:
                new_qhist /= self.bin_width
        return ChargeHistogram(self.qcenter, new_qhist, self.bin_width)

    def to_log10(self, bins_per_decade=10, density=False):
        """
        Transforms the histogram into log10 space.

        Args:
            bins_per_decade (int): Number of bins per decade in log10 space.
            density (bool): If True, the resulting histogram will be a density
                            (divided by the log bin width).

        Returns:
            ChargeHistogram: A new histogram with log10(q) as bin centers.
        """
        # Filter positive qcenters
        mask = self.qcenter > 0
        if not np.any(mask):
            # Return an empty histogram if no positive values
            return ChargeHistogram(np.array([0.0]),
                                   np.zeros(self.shape + (1,)),
                                   1.0 / bins_per_decade)

        q_pos = self.qcenter[mask]
        log_q = np.log10(q_pos)

        log_bin_width = 1.0 / bins_per_decade

        # Define log10 edges covering the range of positive qcenters
        min_log = np.floor(np.min(log_q) * bins_per_decade) / bins_per_decade
        max_log = np.ceil(np.max(log_q) * bins_per_decade) / bins_per_decade
        log_edges = np.arange(min_log, max_log + 0.5 * log_bin_width, log_bin_width)

        if len(log_edges) < 2:
             log_edges = np.array([min_log, min_log + log_bin_width])

        new_qcenter = 0.5 * (log_edges[:-1] + log_edges[1:])

        # Original edges and CDF
        edges_orig, cdf_orig = self.cdf()

        # Interpolate CDF at target edges (in linear space)
        target_edges = 10**log_edges

        grid_shape = self.shape
        num_elements = int(np.prod(grid_shape)) if grid_shape else 1

        # Flatten for interpolation
        flat_cdf = cdf_orig.reshape(num_elements, -1)
        new_flat_cdf = np.zeros((num_elements, len(log_edges)))

        for i in range(num_elements):
            new_flat_cdf[i] = np.interp(target_edges, edges_orig, flat_cdf[i])

        # Probability in each log bin
        new_qhist_flat = np.diff(new_flat_cdf, axis=-1)

        # Multiply by original total sum to preserve scale (e.g. counts)
        total_sum = np.sum(self.qhist, axis=-1).reshape(num_elements, 1)
        new_qhist_flat *= total_sum

        if density:
            new_qhist_flat /= log_bin_width

        new_qhist = new_qhist_flat.reshape(grid_shape + (len(new_qcenter),))

        return ChargeHistogram(new_qcenter, new_qhist, log_bin_width)

    def cdf(self, reverse=False):
        """
        Returns the normalized cumulative distribution function (CDF) for all elements.

        Args:
            reverse (bool): If True, returns the reversed CDF (1 to 0),
                            representing the probability P(X >= x).

        Returns:
            tuple: (edges, cdf)
                edges: np.ndarray of bin edges (length N+1)
                cdf: np.ndarray of normalized cumulative counts (shape: grid_shape + (N+1))
        """
        # If qcenter are centers, left edges are qcenter - 0.5 * bin_width
        left_edges = self.qcenter - 0.5 * self.bin_width
        edges = np.append(left_edges, left_edges[-1] + self.bin_width)

        total_counts = np.sum(self.qhist, axis=-1)

        # Result CDF array
        cdf_shape = self.qhist.shape[:-1] + (self.qhist.shape[-1] + 1,)
        cdf = np.zeros(cdf_shape)

        good_mask = total_counts > 0
        if not good_mask.any():
            return edges, cdf

        if reverse:
            # P(X >= x)
            # cumsum from the right
            cumsum = np.cumsum(self.qhist[..., ::-1], axis=-1)[..., ::-1]
            cdf[good_mask, :-1] = cumsum[good_mask] / total_counts[good_mask, np.newaxis]
            # cdf[..., -1] is already 0
        else:
            # P(X <= x)
            # Cumulative sum along the range axis
            cumsum = np.cumsum(self.qhist, axis=-1)
            cdf[good_mask, 1:] = cumsum[good_mask] / total_counts[good_mask, np.newaxis]
            # cdf[..., 0] is already 0

        return edges, cdf

    def sf(self):
        """
        Returns the survival function (SF) for all elements in the histogram grid.
        SF is defined as 1 - CDF, representing P(X >= x).

        Returns:
            tuple: (edges, sf)
                edges: np.ndarray of bin edges (length N+1)
                sf: np.ndarray of normalized survival function values (shape: grid_shape + (N+1))
        """
        return self.cdf(reverse=True)

    def quantiles(self, q):
        """
        Computes the quantiles for all elements in the histogram grid.

        Args:
            q (float or iterable): The quantile(s) to compute (e.g., 0.5 for median).

        Returns:
            list of np.ndarray: A list containing an array of quantile values for each q.
        """
        # Ensure q is an array
        q_vals = np.atleast_1d(q)

        edges, cdf = self.cdf()

        # Flatten for interpolation
        flat_cdf = cdf.reshape(-1, cdf.shape[-1])
        grid_shape = self.shape
        num_elements = flat_cdf.shape[0]

        # Mask for histograms with data
        # Last element of CDF is 1.0 for those with data
        good_mask = flat_cdf[:, -1] > 0

        results = []
        for qv in q_vals:
            # Output array for this quantile
            res = np.full(num_elements, np.nan)

            # For each histogram, interpolate qv
            for i in range(num_elements):
                if good_mask[i]:
                    res[i] = np.interp(qv, flat_cdf[i], edges)

            results.append(res.reshape(grid_shape))

        return results[0] if np.isscalar(q) else results

    def mean(self):
        """Returns the weighted mean for all elements in the histogram grid."""
        total = np.sum(self.qhist, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.sum(self.qhist * self.qcenter, axis=-1) / total

    def var(self):
        """Returns the weighted variance for all elements in the histogram grid."""
        total = np.sum(self.qhist, axis=-1)
        m = self.mean()
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.sum(self.qhist * (self.qcenter - m[..., np.newaxis])**2, axis=-1) / total

    def std(self):
        """Returns the weighted standard deviation for all elements in the histogram grid."""
        return np.sqrt(self.var())

    def median(self):
        """Returns the median for all elements in the histogram grid."""
        return self.quantiles(0.5)

    def iqr(self):
        """Returns the Interquartile Range (IQR) for all elements in the histogram grid."""
        q1, q3 = self.quantiles([0.25, 0.75])
        return q3 - q1

    def winsorized_mean(self, limits=(0.05, 0.05)):
        """
        Computes the Winsorized mean for all elements in the histogram grid.

        Args:
            limits (tuple): The fraction to cut from the (low, high) ends.
        """
        v_low, v_high = self.quantiles([limits[0], 1.0 - limits[1]])
        total = np.sum(self.qhist, axis=-1)

        # Clip bin centers to the Winsorized limits for each grid point
        centers_clipped = np.clip(self.qcenter, v_low[..., np.newaxis], v_high[..., np.newaxis])

        with np.errstate(divide='ignore', invalid='ignore'):
            return np.sum(self.qhist * centers_clipped, axis=-1) / total

    def winsorized_var(self, limits=(0.05, 0.05)):
        """
        Computes the Winsorized variance for all elements in the histogram grid.

        Args:
            limits (tuple): The fraction to cut from the (low, high) ends.
        """
        v_low, v_high = self.quantiles([limits[0], 1.0 - limits[1]])
        total = np.sum(self.qhist, axis=-1)

        wm = self.winsorized_mean(limits=limits)
        centers_clipped = np.clip(self.qcenter, v_low[..., np.newaxis], v_high[..., np.newaxis])

        with np.errstate(divide='ignore', invalid='ignore'):
            return np.sum(self.qhist * (centers_clipped - wm[..., np.newaxis])**2, axis=-1) / total

    def huber_location(self, loss='huber', scale=None, loss_parameter=1.0):
        """
        Estimates the location (center) of the distribution using a robust loss function.

        Args:
            loss (str): The loss function to use ('huber', 'soft_l1', 'cauchy', 'arctan').
                        Defaults to 'huber'.
            scale (array-like, optional): The scale parameter for the loss function.
                                          Defaults to the square root of the Winsorized variance.

        Returns:
            np.ndarray: The estimated location for each point in the grid.
        """
        from scipy.optimize import minimize_scalar

        grid_shape = self.shape
        num_elements = int(np.prod(grid_shape)) if grid_shape else 1

        # Flatten for iteration
        flat_qhist = self.qhist.reshape(num_elements, -1)

        # Get initial guesses and scales
        x0_all = np.atleast_1d(self.median()).flatten()
        if scale is None:
            # Use a combination of IQR and Winsorized variance for scale
            # IQR / 1.349 is a robust estimate of sigma for Gaussian
            iqr_all = np.atleast_1d(self.iqr()).flatten() / 1.349
            wvar_all = np.sqrt(np.atleast_1d(self.winsorized_var())).flatten()

            # Take the smaller of the two, but ensure it's not zero
            scale_all = np.where((iqr_all > 0) & (iqr_all < wvar_all), iqr_all, wvar_all)
        else:
            scale_all = np.asanyarray(scale).flatten()
            if scale_all.size == 1:
                scale_all = np.full(num_elements, scale_all[0])

        # Loss functions rho(z) where z = (x - mu) / scale
        loss_functions = {
            'huber':   lambda z: np.where(np.abs(z) <= loss_parameter, 0.5 * z**2, loss_parameter * np.abs(z) - 0.5 * loss_parameter**2),
            'soft_l1': lambda z: 2 * (np.sqrt(1 + 0.5 * z**2) - 1),
            'cauchy':  lambda z: np.log(1 + 0.5 * z**2),
            'arctan':  lambda z: np.arctan(0.5 * z**2),
            'linear':  lambda z: 0.5 * z**2
        }

        if not loss:
            loss = 'linear'

        if loss not in loss_functions:
            raise ValueError(f"Unknown loss function: {loss}. Supported: {list(loss_functions.keys())}")

        rho = loss_functions[loss]

        results = np.full(num_elements, np.nan)

        for i in range(num_elements):
            s_val = scale_all[i]
            if np.isnan(s_val) or s_val <= 0:
                s_val = self.bin_width

            x0 = x0_all[i]
            if np.isnan(x0):
                continue

            mask = flat_qhist[i] > 0
            if not np.any(mask):
                continue

            w = flat_qhist[i, mask]
            c = self.qcenter[mask]

            def objective(mu):
                return np.sum(w * rho((c - mu) / s_val))

            try:
                # Bracket around the median
                res = minimize_scalar(objective, bracket=(x0 - 5*s_val, x0, x0 + 5*s_val))
                results[i] = res.x
            except Exception:
                results[i] = x0

        return results.reshape(grid_shape) if grid_shape else results[0]

    def huber_scale(self, loss='huber', location=None, loss_parameter=1.0):
        """
        Estimates the scale (standard deviation) of the distribution using a robust loss function.
        The result is calibrated to be consistent with the standard deviation for a Gaussian.

        Args:
            loss (str): The loss function to use ('huber', 'soft_l1', 'cauchy').
                        Defaults to 'huber'.
            location (array-like, optional): The location (center) parameter for the loss function.
                                             Defaults to the median.

        Returns:
            np.ndarray: The estimated scale for each point in the grid.
        """
        from scipy.optimize import minimize_scalar

        grid_shape = self.shape
        num_elements = int(np.prod(grid_shape)) if grid_shape else 1

        # Flatten for iteration
        flat_qhist = self.qhist.reshape(num_elements, -1)

        # Get initial locations and guesses for scale
        if location is None:
            mu_all = np.atleast_1d(self.median()).flatten()
        else:
            mu_all = np.asanyarray(location).flatten()
            if mu_all.size == 1:
                mu_all = np.full(num_elements, mu_all[0])

        # Robust initial scale guess
        iqr_all = np.atleast_1d(self.iqr()).flatten() / 1.349
        wvar_all = np.sqrt(np.atleast_1d(self.winsorized_var())).flatten()
        s0_all = np.where((iqr_all > 0) & (iqr_all < wvar_all), iqr_all, wvar_all)

        loss_functions = {
            'huber':   lambda z: np.where(np.abs(z) <= loss_parameter, 0.5 * z**2, loss_parameter * np.abs(z) - 0.5 * loss_parameter**2),
            'soft_l1': lambda z: 2 * (np.sqrt(1 + 0.5 * z**2) - 1),
            'cauchy':  lambda z: np.log(1 + 0.5 * z**2),
            'linear':  lambda z: 0.5 * z**2
        }

        from scipy.special import erf
        # Calibration constants beta = E[psi(Z)*Z] where Z ~ N(0, 1)
        # These constants ensure the estimator is consistent with Gaussian sigma.
        beta_map = {
            'huber':   erf(loss_parameter / np.sqrt(2.0)), # norm.cdf(k) - norm.cdf(-k)
            'soft_l1': 0.6808803445892556,
            'cauchy':  0.4842562477382487,
            'linear':  1.0
        }

        if not loss:
            loss = 'linear'

        if loss not in loss_functions:
            raise ValueError(f"Unknown loss function for scale estimation: {loss}. "
                             f"Supported: {list(loss_functions.keys())}")

        rho = loss_functions[loss]
        beta = beta_map[loss]

        results = np.full(num_elements, np.nan)

        for i in range(num_elements):
            s0 = s0_all[i]
            if np.isnan(s0) or s0 <= 0:
                s0 = self.bin_width

            mu = mu_all[i]
            if np.isnan(mu):
                continue

            mask = flat_qhist[i] > 0
            if not np.any(mask):
                continue

            w = flat_qhist[i, mask]
            c = self.qcenter[mask]
            w_sum = np.sum(w)

            # The objective function minimizes J(s) = sum(w * rho((c - mu) / s)) + beta * sum(w) * log(s)
            def objective(s):
                if s <= 0:
                    return np.inf
                return np.sum(w * rho((c - mu) / s)) + beta * w_sum * np.log(s)

            try:
                # Bounded optimization is more reliable for scale
                res = minimize_scalar(objective, bounds=(1e-6 * s0, 100 * s0), method='bounded')
                results[i] = res.x
            except Exception:
                results[i] = s0

        return results.reshape(grid_shape) if grid_shape else results[0]

    def huber_location_and_scale(self, loss='huber', loss_parameter=1.0):
        """
        Estimates the location (center) and scale (standard deviation) of the distribution
        simultaneously using a robust loss function.
        The scale result is calibrated to be consistent with the standard deviation for a Gaussian.

        Args:
            loss (str): The loss function to use ('huber', 'soft_l1', 'cauchy').
                        Defaults to 'huber'.

        Returns:
            tuple: (locations, scales)
                locations (np.ndarray): The estimated location for each point in the grid.
                scales (np.ndarray): The estimated scale for each point in the grid.
        """
        from scipy.optimize import minimize

        grid_shape = self.shape
        num_elements = int(np.prod(grid_shape)) if grid_shape else 1

        # Flatten for iteration
        flat_qhist = self.qhist.reshape(num_elements, -1)

        # Get initial locations and guesses for scale
        mu0_all = np.atleast_1d(self.median()).flatten()

        # Robust initial scale guess
        iqr_all = np.atleast_1d(self.iqr()).flatten() / 1.349
        wvar_all = np.sqrt(np.atleast_1d(self.winsorized_var())).flatten()
        s0_all = np.where((iqr_all > 0) & (iqr_all < wvar_all), iqr_all, wvar_all)

        loss_functions = {
            'huber':   lambda z: np.where(np.abs(z) <= loss_parameter, 0.5 * z**2, loss_parameter * np.abs(z) - 0.5 * loss_parameter**2),
            'soft_l1': lambda z: 2 * (np.sqrt(1 + 0.5 * z**2) - 1),
            'cauchy':  lambda z: np.log(1 + 0.5 * z**2),
            'linear':  lambda z: 0.5 * z**2
        }

        from scipy.special import erf
        # Calibration constants beta = E[psi(Z)*Z] where Z ~ N(0, 1)
        # These constants ensure the estimator is consistent with Gaussian sigma.
        beta_map = {
            'huber':   erf(loss_parameter / np.sqrt(2.0)), # norm.cdf(k) - norm.cdf(-k)
            'soft_l1': 0.6808803445892556,
            'cauchy':  0.4842562477382487,
            'linear':  1.0
        }

        if not loss:
            loss = 'linear'

        if loss not in loss_functions:
            raise ValueError(f"Unknown loss function for joint estimation: {loss}. "
                             f"Supported: {list(loss_functions.keys())}")

        rho = loss_functions[loss]
        beta = beta_map[loss]

        results_mu = np.full(num_elements, np.nan)
        results_s = np.full(num_elements, np.nan)

        for i in range(num_elements):
            s0 = s0_all[i]
            if np.isnan(s0) or s0 <= 0:
                s0 = self.bin_width
            
            mu0 = mu0_all[i]
            if np.isnan(mu0):
                continue

            mask = flat_qhist[i] > 0
            if not np.any(mask):
                continue

            w = flat_qhist[i, mask]
            c = self.qcenter[mask]
            w_sum = np.sum(w)

            # The objective function minimizes J(mu, s) = sum(w * rho((c - mu) / s)) + beta * sum(w) * log(s)
            def objective(params):
                mu, s = params
                if s <= 0:
                    return np.inf
                return np.sum(w * rho((c - mu) / s)) + beta * w_sum * np.log(s)

            try:
                res = minimize(objective, x0=[mu0, s0], bounds=[(None, None), (1e-6 * s0, 100 * s0)])
                results_mu[i] = res.x[0]
                results_s[i] = res.x[1]
            except Exception:
                results_mu[i] = mu0
                results_s[i] = s0

        res_mu = results_mu.reshape(grid_shape) if grid_shape else results_mu[0]
        res_s = results_s.reshape(grid_shape) if grid_shape else results_s[0]
        return res_mu, res_s
