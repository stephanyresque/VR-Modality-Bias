from __future__ import annotations


def test_package_imports():
    import vr_modality_bias

    assert hasattr(vr_modality_bias, "__version__")
    assert isinstance(vr_modality_bias.__version__, str)


def test_subpackages_import():
    import vr_modality_bias.data  
    import vr_modality_bias.experiment  
    import vr_modality_bias.io  
    import vr_modality_bias.metrics  
    import vr_modality_bias.models  
    import vr_modality_bias.utils  
