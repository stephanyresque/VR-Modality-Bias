from __future__ import annotations


def test_package_imports():
    import vr_modality_bias

    assert hasattr(vr_modality_bias, "__version__")
    assert isinstance(vr_modality_bias.__version__, str)


def test_subpackages_import():
    import vr_modality_bias.data  # noqa: F401
    import vr_modality_bias.experiment  # noqa: F401
    import vr_modality_bias.io  # noqa: F401
    import vr_modality_bias.metrics  # noqa: F401
    import vr_modality_bias.models  # noqa: F401
    import vr_modality_bias.utils  # noqa: F401
