#!/usr/bin/env python3

# __init__.py: pypanodecoder - PANOSETI Science Packet Decoder and Analysis

# Author: Stephen Fegan <sfegan@llr.in2p3.fr> (2026-05-18)
# Laboratoire Leprince-Ringuet, CNRS/IN2P3, Ecole Polytechnique, Institut Polytechnique de Paris

__version__ = '0.1.0'

from .pcapdecoder import (
    PcapDecoder,
    GTIFilter,
    SciencePacket,
    get_panoseti_packets
)

from .eventbuilder import (
    CameraEvent,
    CameraImages,
    CameraEventBuilder,
    get_camera_events,
    load_camera_images,
    load_pcap_camera_images,
    load_pff_camera_images
)

from .data_writer import (
    PcapWriter,
    PffWriter
)

from .sites import (
    SITES,
    SITES_BY_MOD_ID,
    get_site_info
)

from .baseline_database import (
    BaselineDatabase
)

from .charge_histogram import (
    ChargeHistogram
)

from .pedestals import (
    ChargeSpectra,
    calculate_charge_spectra,
    apply_polynomial_pedestal_correction,
    apply_constant_pedestal_correction
)

from .image_cleaning import (
    BaseCleaner,
    PictureBoundaryCleaner,
    IslandCleaner,
    CleaningPipeline,
    clean_picture_boundary,
    clean_islands,
    clean_images,
    invert_cleaned_images
)

from .plotting import (
    plot_image,
    plot_star_field,
    overlay_stars,
    plot_north_west_guide
)

from .stars import (
    get_bright_stars
)

from . import dqm

__all__ = [
    'PcapDecoder',
    'GTIFilter',
    'SciencePacket',
    'get_panoseti_packets',
    'CameraEvent',
    'CameraImages',
    'CameraEventBuilder',
    'get_camera_events',
    'load_camera_images',
    'load_pcap_camera_images',
    'load_pff_camera_images',
    'PcapWriter',
    'PffWriter',
    'SITES',
    'SITES_BY_MOD_ID',
    'get_site_info',
    'BaselineDatabase',
    'ChargeHistogram',
    'ChargeSpectra',
    'calculate_charge_spectra',
    'apply_polynomial_pedestal_correction',
    'apply_constant_pedestal_correction',
    'plot_image',
    'plot_star_field',
    'overlay_stars',
    'plot_north_west_guide',
    'get_bright_stars',
    'dqm',
    'BaseCleaner',
    'PictureBoundaryCleaner',
    'IslandCleaner',
    'CleaningPipeline',
    'clean_picture_boundary',
    'clean_islands',
    'clean_images',
    'invert_cleaned_images'
]
