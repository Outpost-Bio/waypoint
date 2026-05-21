__version__ = "0.1.0"

# Lazy re-exports: `from waypoint_bio import TaxonomicTokenizer` works without
# eagerly importing torch / transformers on every `waypoint --help`.

_LAZY_ATTRS = {
    "TaxonomicTokenizer":             "waypoint_bio.tokenizer",
    "load_tokenizer":                 "waypoint_bio.tokenizer",
    "MicrobiomePretrainingDataset":   "waypoint_bio.dataset",
    "MicrobiomeBenchmarkDataset":     "waypoint_bio.dataset",
    "load_waypoint_dataframe":        "waypoint_bio.dataset",
    "load_abundance_matrix":          "waypoint_bio.abundance_matrix",
    "matrix_to_waypoint_df":          "waypoint_bio.abundance_matrix",
}

__all__ = ["__version__", *sorted(_LAZY_ATTRS)]


def __getattr__(name):
    if name in _LAZY_ATTRS:
        import importlib
        module = importlib.import_module(_LAZY_ATTRS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'waypoint_bio' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
