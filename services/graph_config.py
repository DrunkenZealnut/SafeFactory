"""Domain-specific Knowledge Graph configuration."""

GRAPH_CONFIG = {
    'semiconductor-v2': {
        'enabled': True,
        'hop_depth': 2,
        'graph_weight': 0.3,
        'max_graph_results': 10,
        'entity_types': ['공정', '장비', '물질', '안전규정'],
    },
    'laborlaw': {
        'enabled': True,
        'hop_depth': 1,
        'graph_weight': 0.2,
        'max_graph_results': 8,
        'entity_types': ['법률조항', '권리의무', '절차'],
    },
    'kosha': {
        'enabled': True,
        'hop_depth': 2,
        'graph_weight': 0.3,
        'max_graph_results': 10,
        'entity_types': ['안전규정', '위험요인', '보호장비'],
    },
    'msds': {
        'enabled': False,
    },
    'field-training': {
        'enabled': True,
        'hop_depth': 1,
        'graph_weight': 0.2,
        'max_graph_results': 8,
        'entity_types': ['안전규정', '장비', '절차'],
    },
}


def get_graph_config(namespace: str) -> dict:
    """Return graph config for the given namespace."""
    return GRAPH_CONFIG.get(namespace, {'enabled': False})
