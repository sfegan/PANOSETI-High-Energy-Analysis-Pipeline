#!/usr/bin/env python3

# pedestals.py: Calculate charge spectra and pedestal values

# Author: Stephen Fegan <sfegan@llr.in2p3.fr> (2026-05-13)
# Laboratoire Leprince-Ringuet, CNRS/IN2P3, Ecole Polytechnique, Institut Polytechnique de Paris


import time
import numpy as np

from .eventbuilder import CameraImages
from .baseline_database import BaselineDatabase
from .charge_histogram import ChargeHistogram

__all__ = [
    'BaselineDatabase',
    'ChargeHistogram',
    'ChargeSpectra',
    'calculate_charge_spectra',
    'apply_polynomial_pedestal_correction',
    'apply_constant_pedestal_correction',
    'calculate_spline_location_and_scale',
    'apply_precomputed_spline_pedestal_correction',
    'apply_spline_pedestal_correction'
]


class ChargeSpectra:
    """
    Container for PANOSETI charge spectra (pedestal histograms).
    Allows access via attributes (e.g., .pix) or keys (e.g., ['pix']).
    """
    def __init__(self, num_events, pix, sipm, quabo, camera):
        self.num_events = num_events
        self.pix = pix
        self.sipm = sipm
        self.quabo = quabo
        self.camera = camera

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return f"<ChargeSpectra events={self.num_events}>"

    def normalize(self, density=True):
        """
        Normalizes all histograms in the container.

        Args:
            density (bool): If True, normalizes to a density (integral is 1).
                            If False, normalizes so the sum is 1.
        """
        return ChargeSpectra(
            self.num_events,
            self.pix.normalize(density=density),
            self.sipm.normalize(density=density),
            self.quabo.normalize(density=density),
            self.camera.normalize(density=density)
        )

    def to_log10(self, bins_per_decade=10, density=False):
        """
        Transforms all histograms in the container into log10 space.

        Args:
            bins_per_decade (int): Number of bins per decade in log10 space.
            density (bool): If True, the resulting histograms will be densities.

        Returns:
            ChargeSpectra: A new container with log-transformed histograms.
        """
        return ChargeSpectra(
            self.num_events,
            self.pix.to_log10(bins_per_decade=bins_per_decade, density=density),
            self.sipm.to_log10(bins_per_decade=bins_per_decade, density=density),
            self.quabo.to_log10(bins_per_decade=bins_per_decade, density=density),
            self.camera.to_log10(bins_per_decade=bins_per_decade, density=density)
        )

    def downsample(self, factor):
        """
        Downsamples all histograms in the container.

        Args:
            factor (int): The number of bins to combine.

        Returns:
            ChargeSpectra: A new container with downsampled histograms.
        """
        return ChargeSpectra(
            self.num_events,
            self.pix.downsample(factor),
            self.sipm.downsample(factor),
            self.quabo.downsample(factor),
            self.camera.downsample(factor)
        )

def _build_spectra(images, qmin_pix=-512, qmax_pix=4096, downsample=False):
    """Internal helper to build spectra from a 3D numpy array of images (32, 32, N)."""

    # Bin widths
    w_pix = 1
    w_sipm = 8 if downsample else 1
    w_quabo = 16 if downsample else 1
    w_camera = 32 if downsample else 1

    # Histogram for charge each pixel (32x32)
    # Centers are the expected integer values
    qcenter_pix = np.arange(qmin_pix, qmax_pix, w_pix)
    qhist_pix = np.zeros((32, 32, len(qcenter_pix)), dtype=int)

    # Histograms for (summed) charge in each SiPM
    qcenter_sipm = np.arange(qmin_pix*64, qmax_pix*64, w_sipm)
    qhist_sipm = np.zeros((4, 4, len(qcenter_sipm)), dtype=int)

    # Histograms for (summed) charge in each Quabo
    qcenter_quabo = np.arange(qmin_pix*256, qmax_pix*256, w_quabo)
    qhist_quabo = np.zeros((2, 2, len(qcenter_quabo)), dtype=int)

    # Histograms for (summed) charge in the full camera
    qcenter_camera = np.arange(qmin_pix*1024, qmax_pix*1024, w_camera)
    qhist_camera = np.zeros(len(qcenter_camera), dtype=int)

    num_events = images.shape[2]

    ii_pix, jj_pix = np.meshgrid(range(32),range(32))
    ii_sipm, jj_sipm = np.meshgrid(range(4),range(4))
    ii_quabo, jj_quabo = np.meshgrid(range(2),range(2))

    for n in range(num_events):
        image = images[:,:,n]
        valid_pix = ~np.isnan(image)

        # Index: (val - center_0) / w + 0.5. Since center_0 is qmin, it's (val - qmin) / w + 0.5
        if np.any(valid_pix):
            val_pix = image[valid_pix]
            idx_pix = np.floor((val_pix - qmin_pix) / w_pix + 0.5).astype(int)
            idx_pix = np.clip(idx_pix, 0, len(qcenter_pix) - 1)
            qhist_pix[jj_pix[valid_pix], ii_pix[valid_pix], idx_pix] += 1

        # Replace NaNs with 0 for summing SiPM, Quabo, and Camera charges
        image_clean = np.nan_to_num(image, nan=0.0)
        image_sum = np.zeros((33,33), dtype=float)
        image_sum[1:,1:] = np.cumsum(np.cumsum(image_clean, axis=0), axis=1)

        # Count valid (non-NaN) pixels in each block
        valid_count = np.zeros((33,33), dtype=int)
        valid_count[1:,1:] = np.cumsum(np.cumsum(valid_pix.astype(int), axis=0), axis=1)

        # SiPM level: only record SiPMs that contain at least one valid pixel
        n_valid_sipm = np.diff(np.diff(valid_count[::8,::8], axis=0), axis=1)
        valid_sipm = (n_valid_sipm > 0)
        if np.any(valid_sipm):
            image_sipm = np.diff(np.diff(image_sum[::8,::8], axis=0), axis=1)
            idx_sipm = np.floor((image_sipm[valid_sipm] - qmin_pix*64) / w_sipm + 0.5).astype(int)
            idx_sipm = np.clip(idx_sipm, 0, len(qcenter_sipm) - 1)
            qhist_sipm[jj_sipm[valid_sipm], ii_sipm[valid_sipm], idx_sipm] += 1

        # Quabo level: only record Quabos that contain at least one valid pixel
        n_valid_quabo = np.diff(np.diff(valid_count[::16,::16], axis=0), axis=1)
        valid_quabo = (n_valid_quabo > 0)
        if np.any(valid_quabo):
            image_quabo = np.diff(np.diff(image_sum[::16,::16], axis=0), axis=1)
            idx_quabo = np.floor((image_quabo[valid_quabo] - qmin_pix*256) / w_quabo + 0.5).astype(int)
            idx_quabo = np.clip(idx_quabo, 0, len(qcenter_quabo) - 1)
            qhist_quabo[jj_quabo[valid_quabo], ii_quabo[valid_quabo], idx_quabo] += 1

        # Camera level: only record if camera contains at least one valid pixel
        if valid_count[32,32] > 0:
            image_camera = image_sum[32,32]
            idx_camera = int(np.floor((image_camera - qmin_pix*1024) / w_camera + 0.5))
            idx_camera = np.clip(idx_camera, 0, len(qcenter_camera) - 1)
            qhist_camera[idx_camera] += 1

    return ChargeSpectra(
        num_events=num_events,
        pix=ChargeHistogram(qcenter_pix, qhist_pix, w_pix),
        sipm=ChargeHistogram(qcenter_sipm, qhist_sipm, w_sipm),
        quabo=ChargeHistogram(qcenter_quabo, qhist_quabo, w_quabo),
        camera=ChargeHistogram(qcenter_camera, qhist_camera, w_camera)
    )

def calculate_charge_spectra(camera_images, gti_indexes=None, combine_gtis=True,
                             qmin_pix=-512, qmax_pix=4096, downsample=False,
                             density=False):
    """
    Calculate the charge spectrum for each pixel from a CameraImages object.

    Args:
        camera_images (CameraImages): The loaded camera images and metadata.
        gti_indexes (list, optional): List of GTI indexes to include. If None, all are included.
        combine_gtis (bool): If True, combine all selected GTIs into one spectra object.
                            If False, return a dict of spectra objects keyed by GTI index.
        qmin_pix (int): Minimum charge for pixel histograms.
        qmax_pix (int): Maximum charge for pixel histograms.
        downsample (bool): If True, use larger bins for SiPM, Quabo, and camera histograms.
        density (bool): If True, normalize histograms to a density (integral is 1).

    Returns:
        ChargeSpectra or dict: The calculated spectra.
    """
    if gti_indexes is None:
        mask = np.ones(len(camera_images.gti_indexes), dtype=bool)
    else:
        mask = np.isin(camera_images.gti_indexes, gti_indexes)

    if combine_gtis:
        res = _build_spectra(camera_images.images[:, :, mask],
                             qmin_pix=qmin_pix, qmax_pix=qmax_pix, downsample=downsample)
        if density:
            res = res.normalize(density=True)
        return res
    else:
        results = {}
        unique_gtis = np.unique(camera_images.gti_indexes[mask])
        for gti_idx in unique_gtis:
            gti_mask = (camera_images.gti_indexes == gti_idx)
            res = _build_spectra(camera_images.images[:, :, gti_mask],
                                 qmin_pix=qmin_pix, qmax_pix=qmax_pix, downsample=downsample)
            if density:
                res = res.normalize(density=True)
            results[gti_idx] = res
        return results

def apply_polynomial_pedestal_correction(camera_images, norder=0, quantiles=(0.15, 0.85)):
    """
    Fits a polynomial pedestal model to each pixel and subtracts it.
    The fit is performed independently for each GTI.
    Events outside the specified quantiles are excluded from the fit for robustness.

    Args:
        camera_images (CameraImages): The image container.
        norder (int): The order of the polynomial to fit (0=constant, 1=linear, etc.).
        quantiles (tuple): The (low, high) quantile limits for including data in the fit.

    Returns:
        CameraImages: A new container with pedestal-subtracted images.
    """
    def _correct_gti_images(gti_images):
        # Calculate bounds for fitting (robust estimation of the quiet pedestal)
        charge_spectra = calculate_charge_spectra(gti_images, combine_gtis=True)
        q_low, q_high = charge_spectra.pix.quantiles(quantiles)

        # Convert gti_pcap_times from ns to seconds for numerical stability in polyfit
        t = gti_images.gti_pcap_times / 1e9
        pcorr = np.zeros((32, 32, norder + 1))

        for i in range(32):
            for j in range(32):
                y = gti_images.images[i, j, :]
                mask = (y >= q_low[i, j]) & (y <= q_high[i, j])

                if np.sum(mask) > norder:
                    pcorr[i, j, :] = np.polyfit(t[mask], y[mask], norder)
                else:
                    # Fallback to simple mean if not enough data for polynomial fit
                    pcorr[i, j, -1] = np.mean(y)

        # Use the class method to apply the subtraction
        return gti_images.apply_pedestal_corrections(pcorr)

    return camera_images.map_gtis(_correct_gti_images)

def apply_constant_pedestal_correction(images, pedestal_calculator=None, **kwargs):
    """
    Calculates a constant pedestal correction value for each pixel and subtracts it.
    The calculation is performed independently for each GTI.

    Args:
        images (CameraImages): The image container.
        pedestal_calculator (str or callable, optional):
            If a string, one of:
                - "quantile": Uses the specified quantile (default 0.5/median).
                - "huber", "cauchy", "soft_l1", "arctan": Uses the corresponding
                  robust location estimator from ChargeHistogram.
            If None, defaults to "quantile".
            If a callable, it should take a CameraImages object and **kwargs
            and return a (32, 32) array of pedestal values.
        **kwargs: Arguments passed to the calculator (e.g., quantile=0.5 or scale=1.0).

    Returns:
        CameraImages: A new container with pedestal-subtracted images.
    """
    if pedestal_calculator is None or pedestal_calculator == "quantile":
        def internal_pedestal_calculator(gti_images, quantile=0.5, **unused_kwargs):
            charge_spectra = calculate_charge_spectra(gti_images, combine_gtis=True)
            return charge_spectra.pix.quantiles(quantile)
        pedestal_calculator = internal_pedestal_calculator
    elif isinstance(pedestal_calculator, str) and pedestal_calculator in ["huber", "cauchy", "soft_l1", "arctan"]:
        loss_name = pedestal_calculator
        def robust_pedestal_calculator(gti_images, **kwargs):
            charge_spectra = calculate_charge_spectra(gti_images, combine_gtis=True)
            return charge_spectra.pix.huber_location(loss=loss_name, **kwargs)
        pedestal_calculator = robust_pedestal_calculator

    def _correct_gti_images(gti_images):
        pedestal_val = pedestal_calculator(gti_images, **kwargs)
        pcorr = pedestal_val[:, :, np.newaxis]
        return gti_images.apply_pedestal_corrections(pcorr)

    return images.map_gtis(_correct_gti_images)

def calculate_spline_location_and_scale(camera_images, dtknot=600, nknot=None, loss='huber',
                                        max_iter=20, tol=1e-6, ridge=1e-8, loud=False, profile=False, loss_parameter=1.0):
    """
    Estimates the time-dependent location (center) and scale (standard deviation) of the
    distribution simultaneously using a robust loss function assuming a spline for location.
    The scale result is calibrated to be consistent with the standard deviation for a Gaussian.

    Performance notes vs. the scipy.optimize.minimize version:
      1. The cubic-spline basis matrix B (mu_interp = B @ knots) depends only on tknot_s and
         times_s, which are identical for every pixel, so it is built once instead of being
         reconstructed (as a CubicSpline object) on every objective evaluation of every pixel.
      2. The per-pixel optimization is solved via Iteratively Reweighted Least Squares (IRLS):
         each iteration is a weighted linear solve for the knot values (closed form) plus a 1D
         root-find for the scale, instead of a generic bounded quasi-Newton search with
         finite-difference gradients over (nknot + 1) parameters.
      3. All pixels are solved *simultaneously* rather than in a per-pixel Python loop:
         - The knot normal-equations (A, b) are built with two batched GEMMs using a
           precomputed pairwise-basis tensor G = einsum('ti,tj->tij', B, B), and solved with
           a single batched np.linalg.solve call instead of ~num_elements small (nknot,nknot)
           solves.
         - The scale root-find (previously a per-pixel scipy.optimize.brentq call) is replaced
           with a vectorized bisection over all active pixels at once. Bisection needs more
           iterations than Brent's method for the same tolerance, but collapsing ~1e6 tiny
           scalar calls (brentq's internal per-pixel evaluations) into ~60 calls on full-size
           arrays is a large net win at these array sizes. If the scale solve is later shown
           (via profiling) to dominate wall-clock, replacing bisection with a vectorized
           Illinois-modified regula falsi would cut ~60 iterations down to ~15-20.
         - Pixels are dropped out of the active set as soon as they converge, so later
           iterations only pay for pixels that still need work.

    Caveat: for convex losses (huber, soft_l1, linear) IRLS is a majorize-minimize algorithm
    and is guaranteed to decrease the same objective monotonically to a stationary point, so
    results should match the old optimizer to numerical precision. 'cauchy' is a non-convex
    (redescending) loss, so IRLS and BFGS can in principle converge to different local minima
    depending on the starting point -- worth spot-checking a subset of pixels against the old
    implementation before fully switching over, especially for 'cauchy'.

    Args:
        camera_images: images (CameraImages): The image container.
        dtknot: maximum time between knots
        nknot: force number of knots, overriding tknot if not None
        loss (str): The loss function to use ('huber', 'soft_l1', 'cauchy', 'linear').
                    Defaults to 'huber'.
        max_iter (int): maximum number of IRLS iterations per pixel.
        tol (float): relative convergence tolerance on knot values and scale between
                     IRLS iterations.
        ridge (float): small Tikhonov regularization added to the normal equations for
                       numerical stability (e.g. knots with little/no data nearby).

    Returns:
        tuple: (tknots, locations, scales)
            tknot (np.ndarray): Time of the knots. Shape: Nknot
            locations (np.ndarray): The estimated location for each point in the grid. Shape: 32x32xNknot
            scales (np.ndarray): The estimated scale for each point in the grid. Shape: 32x32
    """
    from scipy.interpolate import CubicSpline

    grid_shape = camera_images.images.shape[:-1]
    num_elements = int(np.prod(grid_shape)) if grid_shape else 1

    # Use seconds for knot spacing calculations to make tknot argument units clear
    times_s = camera_images.gti_pcap_times / 1e9
    t_start_s = np.min(times_s)
    t_end_s = np.max(times_s)
    if nknot is None:
        nknot = np.maximum(2, int(np.ceil((t_end_s - t_start_s) / dtknot)) + 1)
    tknot_s = np.linspace(t_start_s, t_end_s, nknot)

    if loud:
        import datetime
        dtknot_actual = int(np.round((t_end_s - t_start_s) / max(1, nknot - 1)))

        if camera_images.pcap_times is not None and len(camera_images.pcap_times) > 0:
            display_t_start_s = np.min(camera_images.pcap_times) / 1e9
        else:
            display_t_start_s = t_start_s

        start_time_str = datetime.datetime.fromtimestamp(display_t_start_s).strftime('%Y-%m-%d %H:%M:%S')
        prefix = f"{start_time_str} {nknot:3d}*{dtknot_actual:3d}s : "

    # Flatten for iteration
    flat_images = camera_images.images.reshape(num_elements, -1)
    n_time = flat_images.shape[1]

    # Get initial locations and robust guesses for scale
    mu0_all = np.median(flat_images, axis=-1)
    s0_all = (np.quantile(flat_images, 0.75, axis=-1) - np.quantile(flat_images, 0.25, axis=-1)) / 1.349

    # Build per-knot initial guesses by taking the median of samples within
    # a time window around each knot (window = +/- dtknot/2 seconds).
    mu0_knots = np.full((num_elements, nknot), np.nan)
    window = float(dtknot) / 2
    if window <= 0:
        total_dur = max(1e-12, (t_end_s - t_start_s))
        window = total_dur / max(2.0 * nknot, 1.0)

    for k in range(nknot):
        t_k = tknot_s[k]
        mask_t = (times_s >= (t_k - window)) & (times_s <= (t_k + window))
        if not np.any(mask_t):
            continue
        vals = flat_images[:, mask_t]
        mu0_knots[:, k] = np.median(vals, axis=-1)

    # Fill any knot windows that had no samples with the pixel's global median.
    nan_mask = np.isnan(mu0_knots)
    mu0_knots = np.where(nan_mask, mu0_all[:, None], mu0_knots)

    # --- Precompute the spline basis matrix once ---
    # CubicSpline(tknot_s, knots)(times_s) is linear in `knots` for fixed tknot_s/times_s,
    # so mu_interp = B @ knots. Build B by evaluating the spline for each unit knot vector,
    # once total, instead of reconstructing a CubicSpline per pixel per objective call.
    B = np.empty((n_time, nknot))
    for k in range(nknot):
        e_k = np.zeros(nknot)
        e_k[k] = 1.0
        B[:, k] = CubicSpline(tknot_s, e_k)(times_s)

    # Pairwise basis products, precomputed once. For a batch of pixels with IRLS weights
    # W (shape P x T), the weighted normal-equation matrices for every pixel in the batch
    # are obtained with a single GEMM: (W @ G).reshape(P, K, K) == B.T @ diag(w_p) @ B per row.
    G = np.einsum('ti,tj->tij', B, B).reshape(n_time, nknot * nknot)

    # IRLS weight functions w(z) = psi(z)/z (finite at z=0 for all four losses below).
    weight_functions = {
        'huber':   lambda z: np.divide(loss_parameter, np.abs(z), out=np.ones_like(z, dtype=float), where=np.abs(z) > loss_parameter),
        'soft_l1': lambda z: 1.0 / np.sqrt(1 + 0.5 * z ** 2),
        'cauchy':  lambda z: 1.0 / (1 + 0.5 * z ** 2),
        'linear':  lambda z: np.ones_like(z),
    }
    
    from scipy.special import erf
    # Calibration constants beta = E[psi(Z)*Z] where Z ~ N(0, 1); same values as the
    # original implementation, since the scale stationarity condition is
    # sum(psi(z)*z) = beta*qlen, i.e. sum(w(z)*z**2) = beta*qlen.
    beta_map = {
        'huber':   erf(loss_parameter / np.sqrt(2.0)),
        'soft_l1': 0.6808803445892556,
        'cauchy':  0.4842562477382487,
        'linear':  1.0,
    }

    if not loss:
        loss = 'linear'
    if loss not in weight_functions:
        raise ValueError(f"Unknown loss function for joint estimation: {loss}. "
                          f"Supported: {list(weight_functions.keys())}")

    wfun = weight_functions[loss]
    beta = beta_map[loss]
    ridge_matrix = ridge * np.eye(nknot)
    qlen = float(n_time)  # every pixel shares the same number of time samples

    def solve_scale_batched(R, s_guess, n_bisect=32, xtol_rel=1e-6):
        """
        Vectorized version of the original per-pixel brentq root-find:
            sum(w(r/s) * (r/s)**2) == beta * qlen  for s
        solved simultaneously for every row of R via bracket expansion + bisection.
        R: (P, T) residuals. s_guess: (P,) previous/initial scale estimate.
        Returns s: (P,) new scale estimate.

        This is the single most expensive call in the whole solve (it runs once, over the
        full active batch, on the first IRLS iteration), since each iteration evaluates
        wfun(z)*z**2 over the entire (P, T) residual array. The bracket spans 1e-3*s_guess
        to 1e3*s_guess, so matching brentq's old xtol=1e-6*s_guess needs log2(1e6/1e-3) ~ 30
        halvings -- not 60. An early-exit on bracket width additionally stops the loop as
        soon as every row is converged, rather than always spending the full budget.
        """
        def g(s):
            z = R / s[:, None]
            return np.sum(wfun(z) * z ** 2, axis=1) - beta * qlen

        lo = s_guess * 1e-3
        hi = s_guess * 1e3
        glo = g(lo)
        ghi = g(hi)

        for _ in range(12):
            bad = glo * ghi > 0
            if not np.any(bad):
                break
            lo = np.where(bad, lo * 0.1, lo)
            hi = np.where(bad, hi * 10.0, hi)
            glo = np.where(bad, g(lo), glo)
            ghi = np.where(bad, g(hi), ghi)

        # Pixels where a bracket still couldn't be found (shouldn't normally happen):
        # keep the previous scale estimate, same fallback as the original implementation.
        unbracketed = glo * ghi > 0

        a, b = lo.copy(), hi.copy()
        for _ in range(n_bisect):
            if np.all((b - a) < xtol_rel * s_guess):
                break
            mid = 0.5 * (a + b)
            gmid = g(mid)
            same_sign = (np.sign(gmid) == np.sign(glo))
            a = np.where(same_sign, mid, a)
            glo = np.where(same_sign, gmid, glo)
            b = np.where(same_sign, b, mid)

        s = 0.5 * (a + b)
        return np.where(unbracketed, s_guess, s)

    def solve_all_pixels(Q, mu0, s0, profile=False):
        """
        Batched IRLS over all valid pixels at once.
        Q: (P, T) data. mu0: (P, K) initial knot values. s0: (P,) initial scale.
        Returns knots (P, K), s (P,), converged (P,) bool, niter (P,) int.
        If profile=True, also returns a dict of cumulative wall-clock seconds spent in
        each phase (gemm_solve, scale_solve_exact, scale_solve_approx), so you can see
        which part actually dominates instead of guessing from FLOP counts.
        """
        P = Q.shape[0]
        knots = mu0.astype(float).copy()
        s = s0.astype(float).copy()
        converged = np.zeros(P, dtype=bool)
        niter = np.zeros(P, dtype=int)
        active = np.ones(P, dtype=bool)

        timings = {'gemm_solve': 0.0, 'scale_solve_exact': 0.0, 'scale_solve_approx': 0.0}

        for it in range(max_iter):
            idx = np.where(active)[0]
            if idx.size == 0:
                break

            Qa = Q[idx]
            Ka = knots[idx]
            Sa = s[idx]

            t0 = time.monotonic() if profile else None

            R = Qa - Ka @ B.T
            Z = R / Sa[:, None]
            W = wfun(Z)

            A = (W @ G).reshape(-1, nknot, nknot) + ridge_matrix
            b_vec = (W * Qa) @ B
            try:
                # NumPy's batched solve requires b to be explicitly (..., M, 1) here --
                # a plain (P, K) array is interpreted as one (P, K) matrix RHS, not a
                # batch of P length-K vectors, and raises a core-dimension mismatch.
                knots_new = np.linalg.solve(A, b_vec[..., None])[..., 0]
            except np.linalg.LinAlgError:
                # Rare: fall back to a per-row lstsq only for the rows that fail.
                knots_new = np.empty_like(b_vec)
                for j in range(A.shape[0]):
                    try:
                        knots_new[j] = np.linalg.solve(A[j], b_vec[j])
                    except np.linalg.LinAlgError:
                        knots_new[j], *_ = np.linalg.lstsq(A[j], b_vec[j], rcond=None)

            if profile:
                t1 = time.monotonic()
                timings['gemm_solve'] += t1 - t0

            R_new = Qa - knots_new @ B.T

            if False: # it == 0:
                s_new = solve_scale_batched(R_new, Sa)
                if profile:
                    t2 = time.monotonic()
                    timings['scale_solve_exact'] += t2 - t1
            else:
                s_step = np.sqrt(np.sum(W * R_new ** 2, axis=1) / (beta * qlen))
                s_new = np.clip(s_step, 0.5 * Sa, 2.0 * Sa)
                if profile:
                    t2 = time.monotonic()
                    timings['scale_solve_approx'] += t2 - t1

            d_knots = np.max(np.abs(knots_new - Ka), axis=1) / np.maximum(1.0, np.max(np.abs(Ka), axis=1))
            d_s = np.abs(s_new - Sa) / Sa

            knots[idx] = knots_new
            s[idx] = s_new
            niter[idx] += 1

            newly_done = (d_knots < tol) & (d_s < tol)
            done_idx = idx[newly_done]
            converged[done_idx] = True
            active[done_idx] = False

            if loud:
                n_active = int(np.sum(active))
                hashes = '#' * min(32, int(32 * (P - n_active) / max(P, 1)))
                print(f"\r{prefix}{hashes:<32} (iter {it+1:2d}/{max_iter}, active {n_active}/{P})",
                      end='', flush=True)

        if profile:
            return knots, s, converged, niter, timings
        return knots, s, converged, niter

    results_mu = np.full((num_elements, nknot), np.nan)
    results_s = np.full(num_elements, np.nan)

    # Only pixels with a sane initial guess are worth solving; everything else stays NaN,
    # matching the original per-pixel skip logic.
    valid = (~np.isnan(s0_all)) & (s0_all > 0) & (~np.isnan(mu0_all))
    valid_idx = np.where(valid)[0]

    num_pix_converged = 0
    num_iter = 0
    start_time = time.monotonic()

    if valid_idx.size > 0:
        Q_valid = flat_images[valid_idx]
        mu0_valid = mu0_knots[valid_idx]
        s0_valid = s0_all[valid_idx]

        try:
            if profile:
                knots_v, s_v, converged_v, niter_v, timings = solve_all_pixels(
                    Q_valid, mu0_valid, s0_valid, profile=True)
                total = sum(timings.values())
                print("\n---- phase breakdown ----")
                for k, v in timings.items():
                    pct = 100 * v / total if total > 0 else 0.0
                    print(f"  {k:20s} {v:8.3f}s  ({pct:5.1f}%)")
                print(f"  {'sum':20s} {total:8.3f}s")
                print("--------------------------")
            else:
                knots_v, s_v, converged_v, niter_v = solve_all_pixels(Q_valid, mu0_valid, s0_valid)
            results_mu[valid_idx, :] = knots_v
            results_s[valid_idx] = s_v
            num_pix_converged = int(np.sum(converged_v))
            num_iter = int(np.sum(niter_v))
        except Exception:
            import traceback
            traceback.print_exc()

    if loud:
        end_time = time.monotonic()
        hashes = '#' * 32
        print(f"\r{prefix}{hashes} (Converged: {num_pix_converged}, Time: {end_time - start_time:.2f}s)", flush=True)

    res_mu = results_mu.reshape(np.append(grid_shape, nknot)) if grid_shape else results_mu[0]
    res_s = results_s.reshape(grid_shape) if grid_shape else results_s[0]
    tknot_ns = (tknot_s * 1e9)
    return tknot_ns, res_mu, res_s

def apply_precomputed_spline_pedestal_correction(camera_images, tknot_ns, knots):
    """
    Applies a precomputed spline pedestal correction to each pixel's image values.
 
    Args:
        camera_images (CameraImages): The image container.
        tknot_ns (np.ndarray): Time of the knots in nanoseconds.
        knots (np.ndarray): 3D array of shape (32, 32, Nknot) containing knot values.

    Returns:
        CameraImages: A new object with adjusted images.
    """
    from scipy.interpolate import CubicSpline
    t_sec = camera_images.gti_pcap_times / 1e9
    tknot_s = tknot_ns / 1e9
    
    cs = CubicSpline(tknot_s, knots, axis=-1)
    res = cs(t_sec)
    
    return CameraImages(
        images=camera_images.images - res,
        event_times=camera_images.event_times,
        pcap_times=camera_images.pcap_times,
        gti_indexes=camera_images.gti_indexes,
        gti_pcap_times=camera_images.gti_pcap_times,
        quabo_masks=camera_images.quabo_masks,
        gtis=camera_images.gtis,
        events=camera_images.events,
        filter=dict(camera_images.filter),
        source=dict(camera_images.source),
        quabo_pcap_time=camera_images.quabo_pcap_time,
        quabo_event_time=camera_images.quabo_event_time
    )

def apply_spline_pedestal_correction(camera_images, dtknot=600, nknot=None, loss='huber',
                                     max_iter=20, tol=1e-6, ridge=1e-8, loud=False, loss_parameter=1.0):
    """
    Calculates a spline pedestal model for each pixel and subtracts it.
    The fit is performed independently for each GTI.

    Args:
        camera_images (CameraImages): The image container.
        dtknot (float): maximum time between knots
        nknot (int): force number of knots, overriding tknot if not None
        loss (str): The loss function to use ('huber', 'soft_l1', 'cauchy', 'linear').
                    Defaults to 'huber'.
        max_iter (int): maximum number of IRLS iterations per pixel.
        tol (float): relative convergence tolerance.
        ridge (float): small Tikhonov regularization.

    Returns:
        CameraImages: A new container with pedestal-subtracted images.
    """
    def _correct_gti_images(gti_images):
        tknot_ns, locations, scales = calculate_spline_location_and_scale(
            gti_images, dtknot=dtknot, nknot=nknot, loss=loss,
            max_iter=max_iter, tol=tol, ridge=ridge, loud=loud, loss_parameter=loss_parameter
        )
        return apply_precomputed_spline_pedestal_correction(gti_images, tknot_ns, locations)

    return camera_images.map_gtis(_correct_gti_images)