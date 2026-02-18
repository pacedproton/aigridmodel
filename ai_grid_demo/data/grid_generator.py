"""IEEE 14-bus grid topology generator."""


def build_ieee_14_grid() -> dict:
    """Return IEEE 14-bus topology as a dictionary.

    The IEEE 14-bus system has:
    - 14 buses (nodes)
    - 20 branches (lines + transformers)
    - 5 generators (including slack at bus 1)
    """
    # IEEE 14-bus line data: (from_bus, to_bus) — 0-indexed
    lines = [
        (0, 1), (0, 4), (1, 2), (1, 3), (1, 4),
        (2, 3), (3, 4), (3, 6), (3, 8), (4, 5),
        (5, 10), (5, 11), (5, 12), (6, 7), (6, 8),
        (7, 8), (8, 9), (9, 10), (11, 12), (12, 13),
    ]

    generators = [
        {'id': 0, 'bus': 0, 'type': 'slack'},
        {'id': 1, 'bus': 1, 'type': 'PV'},
        {'id': 2, 'bus': 2, 'type': 'PV'},
        {'id': 3, 'bus': 5, 'type': 'PV'},
        {'id': 4, 'bus': 7, 'type': 'PV'},
    ]

    limits = {
        f'gen_{g["id"]}': {'min': 0, 'max': 100 if g['id'] < 2 else 50}
        for g in generators
    }

    return {
        'n_buses': 14,
        'n_lines': len(lines),
        'n_generators': len(generators),
        'slack_bus': 0,
        'generators': generators,
        'lines': lines,
        'limits': limits,
    }
