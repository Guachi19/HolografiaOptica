"""Ventana de Python para generar una pelicula holografica de Gabor."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from funciones_opticas import (
    build_coordinates,
    calculate_interference,
    calculate_transmission,
    field_intensity,
    field_phase,
    generate_plane_wave,
    load_and_normalize_image,
    make_object_field,
    save_hologram,
    wave_vector,
)

PIXEL_PITCH_M = 2e-6
WAVELENGTH_M = 532e-9
REFERENCE_AMPLITUDE = 1.0
ANGLE_X_DEG = 3.0
ANGLE_Y_DEG = 0.0
PHASE_RAD = 0.0


class HologramGenerator:
    """Interfaz para elegir una imagen y ajustar t0 y beta."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Generador de holograma de Gabor")
        self.root.geometry("1450x900")

        self.image: np.ndarray | None = None
        self.object_field: np.ndarray | None = None
        self.reference_field: np.ndarray | None = None
        self.interference: np.ndarray | None = None
        self.filename = ""
        self.artists = []
        self.colorbars = []

        self.background = tk.DoubleVar(value=0.1)
        self.sensitivity = tk.DoubleVar(value=0.8)

        toolbar = ttk.Frame(root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="Cargar imagen", command=self.load_image).pack(side=tk.LEFT)
        self.save_button = ttk.Button(
            toolbar, text="Guardar holograma .npz", command=self.save, state=tk.DISABLED
        )
        self.save_button.pack(side=tk.LEFT, padx=(8, 0))
        self.status = tk.StringVar(value="Carga una imagen para comenzar.")
        ttk.Label(toolbar, textvariable=self.status).pack(side=tk.LEFT, padx=14)

        body = ttk.Frame(root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        plot_frame = ttk.Frame(body)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controls = ttk.LabelFrame(body, text="Parametros de la pelicula", padding=12)
        controls.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.figure = Figure(figsize=(12, 7.5), constrained_layout=True)
        self.axes = tuple(self.figure.subplots(2, 3).flat)
        for axis in self.axes:
            axis.axis("off")
        self.axes[0].text(
            0.5,
            0.5,
            "Carga una imagen para ver la formacion del holograma",
            ha="center",
            va="center",
        )
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        navigation = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        navigation.update()
        navigation.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(controls, text="t₀ · Transmision de fondo").pack(anchor=tk.W)
        self.background_scale = tk.Scale(
            controls,
            variable=self.background,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=220,
            command=self.update_transmission,
        )
        self.background_scale.pack(fill=tk.X, pady=(0, 18))
        ttk.Label(controls, text="β · Sensibilidad de la pelicula").pack(anchor=tk.W)
        self.sensitivity_scale = tk.Scale(
            controls,
            variable=self.sensitivity,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=220,
            command=self.update_transmission,
        )
        self.sensitivity_scale.pack(fill=tk.X, pady=(0, 18))
        ttk.Label(
            controls,
            text=(
                "Se mantienen fijos el muestreo y la onda de referencia:\n"
                "λ = 532 nm, R₀ = 1, θx = 3°, θy = 0°,\n"
                "paso = 2 µm, fase = 0°.\n\n"
                "I se normaliza por (1 + R₀)².\n"
                "t(x,y) = t₀ + β I(x,y)."
            ),
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

    def load_image(self) -> None:
        """Carga una imagen y prepara los campos que se usan para grabarla."""
        path = filedialog.askopenfilename(
            title="Selecciona la imagen del objeto",
            filetypes=[
                ("Imagenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.image = load_and_normalize_image(path)
            self.filename = Path(path).name
            x, y = build_coordinates(self.image.shape, PIXEL_PITCH_M)
            self.object_field = make_object_field(self.image)
            self.reference_field = generate_plane_wave(
                x,
                y,
                WAVELENGTH_M,
                REFERENCE_AMPLITUDE,
                ANGLE_X_DEG,
                ANGLE_Y_DEG,
                PHASE_RAD,
            )
            self.interference = calculate_interference(
                self.object_field, self.reference_field, REFERENCE_AMPLITUDE
            )
        except (OSError, ValueError) as error:
            messagebox.showerror("No se pudo cargar la imagen", str(error), parent=self.root)
            return

        self.save_button.configure(state=tk.NORMAL)
        self.status.set(f"{self.filename} · {self.image.shape[1]} × {self.image.shape[0]} px")
        self.draw_panels()

    def update_transmission(self, _value: str | None = None) -> None:
        """Vuelve a dibujar la pelicula cuando cambia uno de sus parametros."""
        if self.interference is not None:
            self.draw_panels()

    def draw_panels(self) -> None:
        """Muestra la imagen, los campos, la interferencia y la pelicula."""
        if (
            self.image is None
            or self.object_field is None
            or self.reference_field is None
            or self.interference is None
        ):
            return
        transmission = calculate_transmission(
            self.interference, self.background.get(), self.sensitivity.get()
        )
        arrays = [
            (self.image, "Imagen original", "gray", (0, 1)),
            (np.abs(self.object_field), "Amplitud del objeto |O|", "gray", (0, 1)),
            (
                field_intensity(self.reference_field),
                "Intensidad de referencia |R|²",
                "magma",
                (0, max(REFERENCE_AMPLITUDE**2, 1e-9)),
            ),
            (field_phase(self.reference_field), "Fase de referencia [rad]", "twilight", (-np.pi, np.pi)),
            (self.interference, "Interferencia normalizada I", "gray", (0, 1)),
            (
                transmission,
                f"Película t=t₀+βI · t₀={self.background.get():.2f}, β={self.sensitivity.get():.2f}",
                "viridis",
                (0, max(float(transmission.max()), 1e-9)),
            ),
        ]
        if self.artists and self.artists[0].get_array().shape == self.image.shape:
            for artist, (array, title, _cmap, limits) in zip(self.artists, arrays):
                artist.set_data(array)
                artist.axes.set_title(title)
                if title.startswith("Película"):
                    artist.set_clim(*limits)
        else:
            for colorbar in self.colorbars:
                colorbar.remove()
            self.colorbars.clear()
            self.artists.clear()
            for axis, (array, title, cmap, limits) in zip(self.axes, arrays):
                axis.clear()
                artist = axis.imshow(
                    array,
                    cmap=cmap,
                    origin="upper",
                    interpolation="nearest",
                    vmin=limits[0],
                    vmax=limits[1],
                )
                axis.set_title(title)
                axis.set_xlabel("x [pixel]")
                axis.set_ylabel("y [pixel]")
                self.artists.append(artist)
                self.colorbars.append(
                    self.figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
                )
        self.canvas.draw_idle()

    def save(self) -> None:
        """Guarda el holograma y los parametros actuales en un archivo .npz."""
        if (
            self.image is None
            or self.object_field is None
            or self.reference_field is None
            or self.interference is None
        ):
            return
        output_directory = Path(__file__).resolve().parent / "hologramas"
        output_directory.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Guardar holograma",
            initialdir=str(output_directory),
            initialfile="holograma_gabor.npz",
            defaultextension=".npz",
            filetypes=[("Archivo NumPy comprimido", "*.npz")],
        )
        if not path:
            return

        metadata = {
            "pixel_pitch_m": PIXEL_PITCH_M,
            "wavelength_m": WAVELENGTH_M,
            "reference_amplitude": REFERENCE_AMPLITUDE,
            "angle_x_deg": ANGLE_X_DEG,
            "angle_y_deg": ANGLE_Y_DEG,
            "phase_rad": PHASE_RAD,
            "background": self.background.get(),
            "sensitivity": self.sensitivity.get(),
            "image_filename": self.filename,
            "intensity_normalization": "(1 + reference_amplitude)^2",
            "intensity_normalization_value": (1.0 + REFERENCE_AMPLITUDE) ** 2,
        }
        try:
            save_hologram(
                path,
                calculate_transmission(
                    self.interference, self.background.get(), self.sensitivity.get()
                ),
                self.interference,
                self.object_field,
                self.reference_field,
                self.image,
                metadata,
            )
        except (OSError, ValueError) as error:
            messagebox.showerror("No se pudo guardar", str(error), parent=self.root)
            return
        self.status.set(f"Holograma guardado: {path}")
        messagebox.showinfo("Guardado", f"Se guardo el holograma en:\n{path}", parent=self.root)


def main() -> None:
    """Crea la ventana principal y ejecuta el bucle de Tkinter."""
    root = tk.Tk()
    HologramGenerator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
