"""Funciones compartidas para la simulacion escalar de holografia de Gabor.

Las longitudes se expresan en metros, las frecuencias espaciales en ciclos/m
y los campos son arrays complejos. La imagen se interpreta como amplitud
transmitida, con fase inicial nula.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError

SCHEMA_VERSION = 1
REQUIRED_HOLOGRAM_ARRAYS = {
    "transmission",
    "interference",
    "object_field",
    "reference_field",
    "original_image",
    "metadata",
}


def load_and_normalize_image(
    source: str | Path | bytes | bytearray | BinaryIO,
    max_size: int = 512,
) -> np.ndarray:
    """Carga una imagen, la convierte a gris, la limita de tamano y escala a [0, 1]."""
    if max_size < 1:
        raise ValueError("max_size debe ser un entero positivo.")
    image_source: Any = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    try:
        with Image.open(image_source) as image:
            image = image.convert("L")
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("No se pudo leer la imagen proporcionada.") from exc

    low = float(np.min(pixels))
    high = float(np.max(pixels))
    if high > low:
        pixels = (pixels - low) / (high - low)
    else:
        pixels = np.zeros_like(pixels)
    return pixels.astype(np.float32, copy=False)


def build_coordinates(
    shape: tuple[int, int], pixel_pitch_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve mallas x,y centradas; la fila superior corresponde a y positivo."""
    if len(shape) != 2 or any(int(size) < 1 for size in shape):
        raise ValueError("shape debe contener dos dimensiones positivas.")
    if not np.isfinite(pixel_pitch_m) or pixel_pitch_m <= 0:
        raise ValueError("El paso de pixel debe ser positivo y finito.")
    rows, columns = shape
    x = (np.arange(columns, dtype=np.float64) - (columns - 1) / 2) * pixel_pitch_m
    y = ((rows - 1) / 2 - np.arange(rows, dtype=np.float64)) * pixel_pitch_m
    return np.meshgrid(x, y)


def wave_vector(
    wavelength_m: float, angle_x_deg: float = 0.0, angle_y_deg: float = 0.0
) -> tuple[float, float, float]:
    """Calcula (kx, ky, kz) usando angulos horizontal y vertical independientes."""
    _validate_wavelength(wavelength_m)
    sin_x = np.sin(np.deg2rad(angle_x_deg))
    sin_y = np.sin(np.deg2rad(angle_y_deg))
    if sin_x**2 + sin_y**2 > 1.0 + 1e-12:
        raise ValueError("Los angulos elegidos no corresponden a una onda propagante.")
    k = 2.0 * np.pi / wavelength_m
    return float(k * sin_x), float(k * sin_y), float(k * np.sqrt(max(0.0, 1 - sin_x**2 - sin_y**2)))


def generate_plane_wave(
    x: np.ndarray,
    y: np.ndarray,
    wavelength_m: float,
    amplitude: float = 1.0,
    angle_x_deg: float = 0.0,
    angle_y_deg: float = 0.0,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """Genera A exp(i(kx*x + ky*y + fase)) sobre las coordenadas dadas."""
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("La amplitud debe ser no negativa y finita.")
    kx, ky, _ = wave_vector(wavelength_m, angle_x_deg, angle_y_deg)
    return (amplitude * np.exp(1j * (kx * x + ky * y + phase_rad))).astype(np.complex64)


def make_object_field(
    amplitude: np.ndarray, phase_rad: np.ndarray | float = 0.0
) -> np.ndarray:
    """Construye O(x,y)=A(x,y) exp(i*phi(x,y))."""
    amplitude_array = np.asarray(amplitude, dtype=np.float32)
    if amplitude_array.ndim != 2 or not np.all(np.isfinite(amplitude_array)):
        raise ValueError("La amplitud del objeto debe ser una matriz 2D finita.")
    if np.any(amplitude_array < 0):
        raise ValueError("La amplitud del objeto no puede ser negativa.")
    return (amplitude_array * np.exp(1j * phase_rad)).astype(np.complex64)


def calculate_interference(
    object_field: np.ndarray, reference_field: np.ndarray, reference_amplitude: float
) -> np.ndarray:
    """Calcula |O+R|^2 y la normaliza por la cota (1+R0)^2, sin alterar O ni R."""
    _validate_matching_fields(object_field, reference_field)
    if not np.isfinite(reference_amplitude) or reference_amplitude < 0:
        raise ValueError("La amplitud de referencia debe ser no negativa y finita.")
    raw_intensity = np.abs(object_field + reference_field) ** 2
    upper_bound = max((1.0 + reference_amplitude) ** 2, np.finfo(np.float32).eps)
    return (raw_intensity / upper_bound).astype(np.float32)


def calculate_transmission(
    interference: np.ndarray, background: float = 0.1, sensitivity: float = 0.8
) -> np.ndarray:
    """Calcula t=t0+beta*I para la irradiancia normalizada del holograma."""
    intensity = np.asarray(interference, dtype=np.float32)
    if intensity.ndim != 2 or not np.all(np.isfinite(intensity)):
        raise ValueError("La interferencia debe ser una matriz 2D finita.")
    if not np.isfinite(background) or not np.isfinite(sensitivity):
        raise ValueError("Los parametros de pelicula deben ser finitos.")
    if background < 0 or sensitivity < 0:
        raise ValueError("La transmision de fondo y la sensibilidad no pueden ser negativas.")
    return (background + sensitivity * intensity).astype(np.float32)


def illuminate_hologram(
    transmission: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    wavelength_m: float,
    amplitude: float = 1.0,
    angle_x_deg: float = 0.0,
    angle_y_deg: float = 0.0,
    phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve la onda plana incidente y U=R_ill*t inmediatamente tras la pelicula."""
    incident = generate_plane_wave(
        x, y, wavelength_m, amplitude, angle_x_deg, angle_y_deg, phase_rad
    )
    transmission_array = np.asarray(transmission)
    if transmission_array.shape != incident.shape:
        raise ValueError("La transmision y las coordenadas deben tener igual dimension.")
    return incident, (incident * transmission_array).astype(np.complex64)


def propagate_angular_spectrum(
    field: np.ndarray,
    wavelength_m: float,
    pixel_pitch_m: float,
    distance_m: float,
    bandlimit: bool = True,
) -> np.ndarray:
    """Propaga un campo con el espectro angular, incluyendo ondas evanescentes.

    Para z >= 0, H=exp(i*z*sqrt(k^2-kx^2-ky^2)); las componentes evanescentes
    decaen exponencialmente. El limite BLAS opcional suprime frecuencias que
    quedarian aliasadas por el ancho finito del plano de calculo.
    """
    _validate_wavelength(wavelength_m)
    if not np.isfinite(pixel_pitch_m) or pixel_pitch_m <= 0:
        raise ValueError("El paso de pixel debe ser positivo y finito.")
    if not np.isfinite(distance_m) or distance_m < 0:
        raise ValueError("La distancia de propagacion debe ser no negativa y finita.")
    source = np.asarray(field, dtype=np.complex64)
    if source.ndim != 2 or min(source.shape) < 1:
        raise ValueError("El campo debe ser una matriz compleja 2D.")

    rows, columns = source.shape
    fx = np.fft.fftfreq(columns, d=pixel_pitch_m)
    fy = np.fft.fftfreq(rows, d=pixel_pitch_m)
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    k = 2.0 * np.pi / wavelength_m
    kx_grid = 2.0 * np.pi * fx_grid
    ky_grid = 2.0 * np.pi * fy_grid
    kz_squared = k**2 - kx_grid**2 - ky_grid**2
    kz = np.sqrt(np.maximum(kz_squared, 0.0))
    transfer = np.exp(1j * distance_m * kz)
    evanescent = kz_squared < 0
    if distance_m > 0 and np.any(evanescent):
        transfer[evanescent] = np.exp(
            -distance_m * np.sqrt(-kz_squared[evanescent])
        )

    if bandlimit and distance_m > 0:
        width_x = columns * pixel_pitch_m
        width_y = rows * pixel_pitch_m
        cutoff_x = 1.0 / (
            wavelength_m * np.sqrt(1.0 + (2.0 * distance_m / width_x) ** 2)
        )
        cutoff_y = 1.0 / (
            wavelength_m * np.sqrt(1.0 + (2.0 * distance_m / width_y) ** 2)
        )
        transfer[(np.abs(fx_grid) > cutoff_x) | (np.abs(fy_grid) > cutoff_y)] = 0.0

    propagated = np.fft.ifft2(np.fft.fft2(source) * transfer)
    return propagated.astype(np.complex64)


def field_amplitude(field: np.ndarray) -> np.ndarray:
    """Obtiene la amplitud no negativa de un campo complejo."""
    return np.abs(field)


def field_phase(field: np.ndarray) -> np.ndarray:
    """Obtiene la fase principal del campo en radianes, en [-pi, pi]."""
    return np.angle(field)


def field_intensity(field: np.ndarray) -> np.ndarray:
    """Obtiene la intensidad |U|^2 sin normalizacion de visualizacion."""
    return np.abs(field) ** 2


def intensity_to_uint8(intensity: np.ndarray) -> np.ndarray:
    """Convierte una intensidad 2D a escala de grises de 8 bits para exportar PNG."""
    values = np.asarray(intensity, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("La intensidad a exportar debe ser una matriz 2D finita y no vacia.")
    if np.any(values < 0):
        raise ValueError("La intensidad no puede ser negativa.")
    maximum = float(np.max(values))
    if maximum == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.rint(np.clip(values / maximum, 0.0, 1.0) * 255).astype(np.uint8)


def reconstruction_components(
    object_field: np.ndarray,
    recording_reference: np.ndarray,
    illumination_field: np.ndarray,
    background: float,
    sensitivity: float,
    intensity_normalization: float = 1.0,
) -> dict[str, np.ndarray]:
    """Descompone U en orden cero, orden de imagen y orden conjugado.

    El orden cero agrupa la transmision de fondo y los terminos no portadores
    beta*(|O|^2+|R_record|^2). Las tres componentes suman exactamente U cuando
    se usa el modelo de pelicula lineal definido en este modulo.
    """
    _validate_matching_fields(object_field, recording_reference)
    _validate_matching_fields(object_field, illumination_field)
    if not np.isfinite(intensity_normalization) or intensity_normalization <= 0:
        raise ValueError("La normalizacion de intensidad debe ser positiva y finita.")
    effective_sensitivity = sensitivity / intensity_normalization
    dc = illumination_field * (
        background + effective_sensitivity * (
            np.abs(object_field) ** 2 + np.abs(recording_reference) ** 2
        )
    )
    image_order = (
        effective_sensitivity
        * illumination_field
        * object_field
        * np.conj(recording_reference)
    )
    conjugate_order = (
        effective_sensitivity
        * illumination_field
        * np.conj(object_field)
        * recording_reference
    )
    return {
        "zero_order": dc.astype(np.complex64),
        "image_order": image_order.astype(np.complex64),
        "conjugate_order": conjugate_order.astype(np.complex64),
    }


def save_hologram(
    destination: str | Path | BinaryIO,
    transmission: np.ndarray,
    interference: np.ndarray,
    object_field: np.ndarray,
    reference_field: np.ndarray,
    original_image: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Guarda el holograma, los campos y metadatos JSON en un .npz comprimido."""
    arrays = {
        "transmission": np.asarray(transmission, dtype=np.float32),
        "interference": np.asarray(interference, dtype=np.float32),
        "object_field": np.asarray(object_field, dtype=np.complex64),
        "reference_field": np.asarray(reference_field, dtype=np.complex64),
        "original_image": np.asarray(original_image, dtype=np.float32),
    }
    shape = arrays["transmission"].shape
    if len(shape) != 2 or any(array.shape != shape for array in arrays.values()):
        raise ValueError("Todos los arrays del holograma deben tener la misma forma 2D.")
    clean_metadata = dict(metadata)
    clean_metadata.update({"schema_version": SCHEMA_VERSION, "shape": list(shape)})
    clean_metadata_json = json.dumps(clean_metadata, ensure_ascii=True)
    np.savez_compressed(
        destination,
        **arrays,
        metadata=np.asarray(clean_metadata_json),
    )


def load_hologram(source: str | Path | bytes | bytearray | BinaryIO) -> dict[str, Any]:
    """Carga y valida un .npz generado por save_hologram, sin permitir objetos pickle."""
    npz_source: Any = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    try:
        with np.load(npz_source, allow_pickle=False) as archive:
            missing = REQUIRED_HOLOGRAM_ARRAYS.difference(archive.files)
            if missing:
                raise ValueError(f"El archivo no contiene los campos requeridos: {sorted(missing)}.")
            result = {key: archive[key] for key in REQUIRED_HOLOGRAM_ARRAYS - {"metadata"}}
            metadata = json.loads(str(archive["metadata"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("El archivo no contiene"):
            raise
        raise ValueError("No se pudo leer un holograma .npz valido.") from exc

    shape = result["transmission"].shape
    if len(shape) != 2 or any(array.shape != shape for array in result.values()):
        raise ValueError("El archivo .npz contiene arrays con dimensiones incompatibles.")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Version de archivo .npz no compatible.")
    if list(shape) != metadata.get("shape"):
        raise ValueError("Las dimensiones guardadas en los metadatos no coinciden.")
    required_metadata = {
        "pixel_pitch_m",
        "wavelength_m",
        "reference_amplitude",
        "angle_x_deg",
        "angle_y_deg",
        "phase_rad",
        "background",
        "sensitivity",
        "intensity_normalization_value",
    }
    missing_metadata = required_metadata.difference(metadata)
    if missing_metadata:
        raise ValueError(
            f"Faltan parametros requeridos en los metadatos: {sorted(missing_metadata)}."
        )
    for key in required_metadata:
        value = metadata[key]
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"El parametro '{key}' debe ser numerico y finito.")
    if metadata["pixel_pitch_m"] <= 0 or metadata["wavelength_m"] <= 0:
        raise ValueError("El paso espacial y la longitud de onda deben ser positivos.")
    if metadata["reference_amplitude"] < 0:
        raise ValueError("La amplitud de referencia guardada no puede ser negativa.")
    if metadata["background"] < 0 or metadata["sensitivity"] < 0:
        raise ValueError("Los parametros de pelicula guardados no pueden ser negativos.")
    if metadata["intensity_normalization_value"] <= 0:
        raise ValueError("La normalizacion de intensidad debe ser positiva.")
    wave_vector(
        metadata["wavelength_m"], metadata["angle_x_deg"], metadata["angle_y_deg"]
    )
    result["metadata"] = metadata
    return result


def _validate_wavelength(wavelength_m: float) -> None:
    """Rechaza longitudes de onda que no sean positivas y finitas."""
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("La longitud de onda debe ser positiva y finita.")


def _validate_matching_fields(first: np.ndarray, second: np.ndarray) -> None:
    """Comprueba que dos campos sean matrices 2D con la misma forma."""
    if np.shape(first) != np.shape(second) or np.ndim(first) != 2:
        raise ValueError("Los campos deben ser matrices 2D con dimensiones iguales.")
