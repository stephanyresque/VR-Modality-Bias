from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vr_modality_bias.data.perturbations import noise_image_uniform


@pytest.fixture
def sample_image() -> Image.Image:
    """A 64 x 32 RGB image with a deterministic pattern (red-dominant)."""
    arr = np.zeros((32, 64, 3), dtype=np.uint8)
    arr[..., 0] = 200
    return Image.fromarray(arr, mode="RGB")


def test_same_seed_produces_same_array(sample_image: Image.Image):
    a = np.array(noise_image_uniform(sample_image, seed=123))
    b = np.array(noise_image_uniform(sample_image, seed=123))
    np.testing.assert_array_equal(a, b)


def test_different_seeds_produce_different_arrays(sample_image: Image.Image):
    a = np.array(noise_image_uniform(sample_image, seed=1))
    b = np.array(noise_image_uniform(sample_image, seed=2))
    assert not np.array_equal(a, b)


def test_shape_and_mode_preserved(sample_image: Image.Image):
    out = noise_image_uniform(sample_image, seed=7)
    assert out.size == sample_image.size  # PIL .size is (W, H)
    assert out.mode == "RGB"
    arr = np.array(out)
    assert arr.shape == (32, 64, 3)  # (H, W, C)
    assert arr.dtype == np.uint8


def test_pixels_cover_full_dynamic_range(sample_image: Image.Image):
    """With ~6144 pixels per channel, both ends of [0, 255] should appear."""
    arr = np.array(noise_image_uniform(sample_image, seed=42))
    assert arr.min() <= 5
    assert arr.max() >= 250


def test_output_is_independent_of_input_pixel_content():
    """The noise must depend on (size, seed) only, never on input pixels."""
    a = np.zeros((32, 64, 3), dtype=np.uint8)
    b = np.full((32, 64, 3), 255, dtype=np.uint8)
    out_a = np.array(noise_image_uniform(Image.fromarray(a, mode="RGB"), seed=99))
    out_b = np.array(noise_image_uniform(Image.fromarray(b, mode="RGB"), seed=99))
    np.testing.assert_array_equal(out_a, out_b)


def test_grayscale_input_yields_three_channel_rgb_output():
    """Inputs in mode "L" must still produce a (H, W, 3) RGB output."""
    gray = Image.fromarray(np.full((20, 30), 128, dtype=np.uint8), mode="L")
    out = noise_image_uniform(gray, seed=0)
    assert out.mode == "RGB"
    assert np.array(out).shape == (20, 30, 3)


def test_works_for_non_square_image(sample_image: Image.Image):
    """Width != height must be handled correctly (PIL size order)."""
    out = noise_image_uniform(sample_image, seed=0)
    assert out.size == (64, 32)  # (W, H)
