#!/usr/bin/env python3

# sites.py: PANOSETI site configurations and module ID mappings.

# Author: Stephen Fegan <sfegan@llr.in2p3.fr> (2026-06-14)
# Laboratoire Leprince-Ringuet, CNRS/IN2P3, Ecole Polytechnique, Institut Polytechnique de Paris

__all__ = [
    'SITES',
    'SITES_BY_MOD_ID',
    'get_site_info'
]

SITES = {
    'gattini': {
        'name': 'gattini',
        'base_ip': '192.168.3.248',
        'daq_ip': '192.168.0.4',
        'scope': 'Gattini',
        'mod_id': 254,
        'module_id': 254
    },
    'winter': {
        'name': 'winter',
        'base_ip': '192.168.3.244',
        'daq_ip': '192.168.0.6',
        'scope': 'Winter',
        'mod_id': 253,
        'module_id': 253
    },
    'fern': {
        'name': 'fern',
        'base_ip': '192.168.3.240',
        'daq_ip': '192.168.0.9',
        'scope': 'Fern',
        'mod_id': 252,
        'module_id': 252
    },
    'pti': {
        'name': 'pti',
        'base_ip': '192.168.3.232',
        'daq_ip': '192.168.0.8',
        'scope': 'PTI',
        'mod_id': 250,
        'module_id': 250
    },
}

SITES_BY_MOD_ID = {info['mod_id']: info for info in SITES.values()}


def get_site_info(module_id):
    """
    Returns site information for a given module ID.
    """
    if module_id is not None and module_id in SITES_BY_MOD_ID:
        res = SITES_BY_MOD_ID[module_id].copy()
        res.setdefault('module_id', module_id)
        return res

    return {
        'scope': 'Unknown',
        'module_id': module_id if module_id is not None else 'unknown',
        'mod_id': module_id if module_id is not None else 'unknown',
        'name': 'unknown',
        'daq_ip': '10.0.1.100',
        'base_ip': '10.0.1.0'
    }
