from __future__ import annotations

import hashlib

from vr_modality_bias.utils.seeds import derive_image_seed


def test_derive_image_seed_is_deterministic():
    a = derive_image_seed(42, "000000000139")
    b = derive_image_seed(42, "000000000139")
    assert a == b


def test_derive_image_seed_distinct_image_ids_yield_distinct_seeds():
    seeds = {derive_image_seed(42, f"000000000{i:03d}") for i in range(50)}
    assert len(seeds) == 50, "unexpected collisions on a small sample"


def test_derive_image_seed_distinct_global_seeds_yield_distinct_seeds():
    seeds = {derive_image_seed(g, "000000000139") for g in range(50)}
    assert len(seeds) == 50


def test_derive_image_seed_fits_in_uint32():
    seed = derive_image_seed(2**31, "any_image_id")
    assert 0 <= seed < 2**32


def test_derive_image_seed_matches_specified_formula():
    """The seed must equal (seed_global + int(sha256(image_id)[:8], 16)) % 2**32."""
    image_id = "deterministic-id"
    expected = (
        42 + int(hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:8], 16)
    ) % (2**32)
    assert derive_image_seed(42, image_id) == expected


def test_derive_image_seed_handles_unicode_image_ids():
    """The seed derivation must not break on non-ASCII image ids."""
    a = derive_image_seed(42, "ímágem-único-🟢")
    b = derive_image_seed(42, "ímágem-único-🟢")
    assert a == b
