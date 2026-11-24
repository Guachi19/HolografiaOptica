"""Ventana de Python para reconstruir un holograma de Gabor guardado en .npz."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import Image

from funciones_opticas import (
    build_coordinates,
    calculate_transmission,
    field_amplitude,
    field_intensity,
    field_phase,
    intensity_to_uint8,
    illuminate_hologram,
    load_hologram,
    propagate_angular_spectrum,
    reconstruction_components,
    save_hologram,
)

OBSERVATION_DISTANCE_M = 10e-3


class HologramReconstructor:
    """Interfaz para ajustar t0 y beta y observar la reconstruccion."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Reconstruccion de holograma de Gabor")
        self.root.geometry("1550x1000")

        self.hologram: dict | None = None
        self.path: str | None = None
        self.colorbars = []
        self.artists = []
        self.update_after_id: str | None = None
        self.reconstructed_intensity: np.ndarray | None = None
        self.background = tk.DoubleVar(value=0.1)
        self.sensitivity = tk.DoubleVar(value=0.8)

        toolbar = ttk.Frame(root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="Cargar holograma .npz", command=self.load).pack(side=tk.LEFT)
        self.save_button = ttk.Button(
            toolbar, text="Guardar con estos parametros", command=self.save, state=tk.DISABLED
        )
        self.save_button.pack(side=tk.LEFT, padx=(8, 0))
        self.save_image_button = ttk.Button(
            toolbar,
            text="Guardar imagen reconstruida",
            command=self.save_reconstructed_image,
            state=tk.DISABLED,
        )
        self.save_image_button.pack(side=tk.LEFT, padx=(8, 0))
        self.status = tk.StringVar(value="Carga el archivo .npz generado por el programa 1.")
        ttk.Label(toolbar, textvariable=self.status).pack(side=tk.LEFT, padx=14)

        body = ttk.Frame(root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        plot_frame = ttk.Frame(body)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controls = ttk.LabelFrame(body, text="Parametros de la pelicula", padding=12)
        controls.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.figure = Figure(figsize=(13, 9), constrained_layout=True)
        self.axes = tuple(self.figure.subplots(3, 3).flat)
        for axis in self.axes:
            axis.axis("off")
        self.axes[0].text(
            0.5,
            0.5,
            "Carga un archivo .npz para iniciar la reconstruccion",
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
            state=tk.DISABLED,
            command=self.schedule_update,
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
            state=tk.DISABLED,
            command=self.schedule_update,
        )
        self.sensitivity_scale.pack(fill=tk.X, pady=(0, 18))
        self.illumination_info = tk.StringVar(value="La onda de grabacion se usa como iluminacion.")
        ttk.Label(
            controls,
            textvariable=self.illumination_info,
            justify=tk.LEFT,
            wraplength=250,
        ).pack(anchor=tk.W, pady=(0, 12))
        ttk.Label(
            controls,
            text=(
                "Distancia de propagacion fija: z = 10 mm.\n\n"
                "U = R_ill · t\n"
                "t = t₀ + β I_norm\n\n"
                "Las intensidades mostradas para los ordenes "
                "separados no se suman entre si."
            ),
            justify=tk.LEFT,
            wraplength=250,
        ).pack(anchor=tk.W)

    def load(self) -> None:
        """Abre un holograma .npz y habilita los controles de reconstruccion."""
        path = filedialog.askopenfilename(
            title="Selecciona el holograma",
            filetypes=[("Hologramas NumPy", "*.npz"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        try:
            hologram = load_hologram(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("No se pudo cargar el holograma", str(error), parent=self.root)
            return

        self.hologram = hologram
        self.path = path
        metadata = hologram["metadata"]
        self.background.set(float(metadata["background"]))
        self.sensitivity.set(float(metadata["sensitivity"]))
        self.background_scale.configure(state=tk.NORMAL)
        self.sensitivity_scale.configure(state=tk.NORMAL)
        self.save_button.configure(state=tk.NORMAL)
        self.save_image_button.configure(state=tk.NORMAL)
        self.illumination_info.set(
            "Iluminacion igual a la referencia de grabacion:\n"
            f"λ = {metadata['wavelength_m'] * 1e9:.0f} nm, "
            f"R₀ = {metadata['reference_amplitude']:.2f}, "
            f"θx = {metadata['angle_x_deg']:.2f}°, "
            f"θy = {metadata['angle_y_deg']:.2f}°."
        )
        self.status.set(f"Cargado: {Path(path).name}")
        self.update_display()

    def schedule_update(self, _value: str | None = None) -> None:
        """Agrupa cambios rapidos de los deslizadores antes de recalcular."""
        if self.hologram is None:
            return
        if self.update_after_id is not None:
            self.root.after_cancel(self.update_after_id)
        self.update_after_id = self.root.after(120, self.update_display)

    def update_display(self) -> None:
        """Aplica la pelicula, propaga el campo y actualiza los paneles."""
        self.update_after_id = None
        if self.hologram is None:
            return
        data = self.hologram
        metadata = data["metadata"]
        transmission = calculate_transmission(
            data["interference"], self.background.get(), self.sensitivity.get()
        )
        x, y = build_coordinates(
            transmission.shape, float(metadata["pixel_pitch_m"])
        )
        incident, emerging = illuminate_hologram(
            transmission,
            x,
            y,
            float(metadata["wavelength_m"]),
            float(metadata["reference_amplitude"]),
            float(metadata["angle_x_deg"]),
            float(metadata["angle_y_deg"]),
            float(metadata["phase_rad"]),
        )
        wavelength = float(metadata["wavelength_m"])
        pixel_pitch = float(metadata["pixel_pitch_m"])
        propagated = propagate_angular_spectrum(
            emerging, wavelength, pixel_pitch, OBSERVATION_DISTANCE_M
        )
        self.reconstructed_intensity = field_intensity(propagated).astype(
            np.float32, copy=False
        )
        components = reconstruction_components(
            data["object_field"],
            data["reference_field"],
            incident,
            self.background.get(),
            self.sensitivity.get(),
            float(metadata["intensity_normalization_value"]),
        )
        propagated_components = {
            name: propagate_angular_spectrum(
                field, wavelength, pixel_pitch, OBSERVATION_DISTANCE_M
            )
            for name, field in components.items()
        }
        panels = [
            (transmission, "Transmision t(x,y)", "gray", None),
            (field_phase(incident), "Fase de iluminacion [rad]", "twilight", (-np.pi, np.pi)),
            (field_amplitude(emerging), "Amplitud despues de la pelicula", "viridis", None),
            (field_phase(emerging), "Fase emergente [rad]", "twilight", (-np.pi, np.pi)),
            (field_intensity(emerging), "Intensidad emergente |U|²", "magma", None),
            (
                self.reconstructed_intensity,
                f"Intensidad propagada, z={OBSERVATION_DISTANCE_M * 1e3:g} mm",
                "magma",
                None,
            ),
            (
                field_intensity(propagated_components["zero_order"]),
                "Orden cero · intensidad individual",
                "magma",
                None,
            ),
            (
                field_intensity(propagated_components["image_order"]),
                "Imagen reconstruida · intensidad",
                "magma",
                None,
            ),
            (
                field_intensity(propagated_components["conjugate_order"]),
                "Imagen conjugada · intensidad",
                "magma",
                None,
            ),
        ]
        self.draw_panels(panels)
        self.status.set(
            f"{Path(self.path).name} · t₀={self.background.get():.2f} · "
            f"β={self.sensitivity.get():.2f}"
        )

    def save_reconstructed_image(self) -> None:
        """Exporta la intensidad reconstruida como PNG o como array NPY."""
        if self.reconstructed_intensity is None or self.path is None:
            return
        source = Path(self.path)
        output_path = filedialog.asksaveasfilename(
            title="Guardar intensidad reconstruida",
            initialdir=str(source.parent),
            initialfile=f"{source.stem}_reconstruida.png",
            defaultextension=".png",
            filetypes=[
                ("Imagen PNG · intensidad normalizada", "*.png"),
                ("Array NumPy · intensidad física", "*.npy"),
            ],
        )
        if not output_path:
            return
        suffix = Path(output_path).suffix.lower()
        try:
            if suffix == ".png":
                Image.fromarray(intensity_to_uint8(self.reconstructed_intensity)).save(
                    output_path
                )
            elif suffix == ".npy":
                with open(output_path, "wb") as output_file:
                    np.save(output_file, self.reconstructed_intensity, allow_pickle=False)
            else:
                raise ValueError("Selecciona el formato PNG o NPY.")
        except (OSError, ValueError) as error:
            messagebox.showerror("No se pudo guardar la reconstruccion", str(error), parent=self.root)
            return
        self.status.set(f"Imagen reconstruida guardada: {output_path}")
        if suffix == ".png":
            details = "PNG en escala de grises, normalizado para visualizacion."
        else:
            details = "Array NPY con la intensidad fisica sin normalizar."
        messagebox.showinfo(
            "Reconstruccion guardada",
            f"{details}\n\nArchivo:\n{output_path}",
            parent=self.root,
        )

    def draw_panels(self, panels: list[tuple]) -> None:
        """Actualiza las imagenes de la ventana y sus barras de color."""
        shape = panels[0][0].shape
        if self.artists and self.artists[0].get_array().shape == shape:
            for artist, (array, title, _cmap, limits) in zip(self.artists, panels):
                artist.set_data(array)
                artist.axes.set_title(title)
                if limits is None:
                    artist.set_clim(0, max(float(np.max(array)), 1e-12))
                else:
                    artist.set_clim(*limits)
        else:
            for colorbar in self.colorbars:
                colorbar.remove()
            self.colorbars.clear()
            self.artists.clear()
            for axis, (array, title, cmap, limits) in zip(self.axes, panels):
                axis.clear()
                kwargs = {}
                if limits is not None:
                    kwargs = {"vmin": limits[0], "vmax": limits[1]}
                artist = axis.imshow(
                    array,
                    cmap=cmap,
                    origin="upper",
                    interpolation="nearest",
                    **kwargs,
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
        """Guarda una copia del holograma con t0 y beta actuales."""
        if self.hologram is None:
            return
        initial = f"{Path(self.path).stem}_ajustado.npz"
        path = filedialog.asksaveasfilename(
            title="Guardar holograma con t₀ y β actuales",
            initialdir=str(Path(self.path).parent),
            initialfile=initial,
            defaultextension=".npz",
            filetypes=[("Hologramas NumPy", "*.npz")],
        )
        if not path:
            return
        data = self.hologram
        metadata = dict(data["metadata"])
        metadata["background"] = self.background.get()
        metadata["sensitivity"] = self.sensitivity.get()
        try:
            save_hologram(
                path,
                calculate_transmission(
                    data["interference"], self.background.get(), self.sensitivity.get()
                ),
                data["interference"],
                data["object_field"],
                data["reference_field"],
                data["original_image"],
                metadata,
            )
        except (OSError, ValueError) as error:
            messagebox.showerror("No se pudo guardar", str(error), parent=self.root)
            return
        self.status.set(f"Guardado: {path}")
        messagebox.showinfo("Guardado", f"Se guardo el holograma en:\n{path}", parent=self.root)


def main() -> None:
    """Crea la ventana principal y ejecuta el bucle de Tkinter."""
    root = tk.Tk()
    HologramReconstructor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
