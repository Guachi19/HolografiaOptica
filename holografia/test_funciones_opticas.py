"""Pruebas unitarias de las funciones opticas y del formato de intercambio."""

import io
import unittest

import numpy as np

from funciones_opticas import (
    build_coordinates,
    calculate_interference,
    calculate_transmission,
    generate_plane_wave,
    intensity_to_uint8,
    load_hologram,
    make_object_field,
    propagate_angular_spectrum,
    reconstruction_components,
    save_hologram,
)


class OpticalFunctionsTests(unittest.TestCase):
    def test_normal_incidence_wave_has_constant_phase_and_amplitude(self):
        x, y = build_coordinates((12, 16), 2e-6)
        wave = generate_plane_wave(x, y, 532e-9, amplitude=1.7)
        np.testing.assert_allclose(np.abs(wave), 1.7, rtol=1e-6)
        np.testing.assert_allclose(np.angle(wave), 0.0, atol=1e-6)

    def test_interference_is_bounded_and_transmission_uses_it(self):
        object_field = make_object_field(np.ones((4, 5), dtype=np.float32))
        reference = np.ones((4, 5), dtype=np.complex64)
        interference = calculate_interference(object_field, reference, 1.0)
        np.testing.assert_allclose(interference, 1.0)
        np.testing.assert_allclose(calculate_transmission(interference, 0.2, 0.5), 0.7)

    def test_intensity_export_scales_for_png_without_mutating_input(self):
        intensity = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        original = intensity.copy()
        exported = intensity_to_uint8(intensity)
        np.testing.assert_array_equal(exported, [[0, 128], [255, 0]])
        np.testing.assert_array_equal(intensity, original)
        np.testing.assert_array_equal(
            intensity_to_uint8(np.zeros((2, 3), dtype=np.float32)),
            np.zeros((2, 3), dtype=np.uint8),
        )

    def test_decomposition_sums_to_the_transmitted_field(self):
        x, y = build_coordinates((16, 20), 2e-6)
        obj = make_object_field(np.linspace(0, 1, 320, dtype=np.float32).reshape(16, 20))
        recording = generate_plane_wave(x, y, 532e-9, 0.8, 4.0, -2.0, 0.3)
        illumination = generate_plane_wave(x, y, 532e-9, 1.1, 4.0, -2.0, 0.3)
        interference = calculate_interference(obj, recording, 0.8)
        background, sensitivity = 0.1, 0.7
        transmission = calculate_transmission(interference, background, sensitivity)
        components = reconstruction_components(
            obj,
            recording,
            illumination,
            background,
            sensitivity,
            intensity_normalization=(1.0 + 0.8) ** 2,
        )
        reconstructed = sum(components.values())
        np.testing.assert_allclose(
            reconstructed, illumination * transmission, rtol=2e-6, atol=2e-6
        )

    def test_angular_spectrum_preserves_a_uniform_field(self):
        field = np.full((32, 24), 1.0 + 0.0j, dtype=np.complex64)
        propagated = propagate_angular_spectrum(field, 532e-9, 2e-6, 0.01)
        np.testing.assert_allclose(np.abs(propagated), 1.0, rtol=1e-6, atol=1e-6)

    def test_npz_round_trip_preserves_fields_and_metadata(self):
        x, y = build_coordinates((8, 10), 2e-6)
        obj = make_object_field(np.ones((8, 10), dtype=np.float32))
        ref = generate_plane_wave(x, y, 532e-9, 1.0, 2.0)
        interference = calculate_interference(obj, ref, 1.0)
        transmission = calculate_transmission(interference, 0.1, 0.8)
        buffer = io.BytesIO()
        save_hologram(
            buffer, transmission, interference, obj, ref, np.ones((8, 10)),
            {
                "pixel_pitch_m": 2e-6,
                "wavelength_m": 532e-9,
                "reference_amplitude": 1.0,
                "angle_x_deg": 2.0,
                "angle_y_deg": 0.0,
                "phase_rad": 0.0,
                "background": 0.1,
                "sensitivity": 0.8,
                "intensity_normalization_value": 4.0,
            },
        )
        loaded = load_hologram(buffer.getvalue())
        np.testing.assert_array_equal(loaded["transmission"], transmission)
        np.testing.assert_array_equal(loaded["object_field"], obj)
        self.assertEqual(loaded["metadata"]["shape"], [8, 10])

    def test_npz_missing_reconstruction_metadata_is_rejected(self):
        x, y = build_coordinates((4, 6), 2e-6)
        obj = make_object_field(np.ones((4, 6), dtype=np.float32))
        ref = generate_plane_wave(x, y, 532e-9)
        interference = calculate_interference(obj, ref, 1.0)
        transmission = calculate_transmission(interference)
        buffer = io.BytesIO()
        save_hologram(
            buffer,
            transmission,
            interference,
            obj,
            ref,
            np.ones((4, 6)),
            {"pixel_pitch_m": 2e-6, "wavelength_m": 532e-9},
        )
        with self.assertRaisesRegex(ValueError, "Faltan parametros requeridos"):
            load_hologram(buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
