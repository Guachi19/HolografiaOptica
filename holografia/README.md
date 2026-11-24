# Simulador interactivo de holografia de Gabor

Dos aplicaciones de escritorio, ejecutables directamente con Python, comparten
`funciones_opticas.py`. Las ventanas se construyen con Tkinter y Matplotlib:

- `01_generar_holograma.py`: carga una imagen, simula la grabacion y exporta un
  archivo `.npz`.
- `02_reconstruir_holograma.py`: abre el `.npz`, ilumina su pelicula y propaga
  el campo mediante el metodo del espectro angular.

## Archivos principales

- `funciones_opticas.py` contiene los calculos, la propagacion y la lectura y
  escritura del formato `.npz`.
- `01_generar_holograma.py` y `02_reconstruir_holograma.py` son las interfaces
  para grabar y reconstruir, respectivamente.
- `test_funciones_opticas.py` comprueba los calculos principales y el intercambio
  de archivos.
- `hologramas/` es la carpeta sugerida para guardar los resultados.

## Instalacion y ejecucion

Instala las dependencias desde esta carpeta. No se necesita Streamlit. En Windows,
Tkinter viene incluido normalmente con la instalacion oficial de Python.

```powershell
python -m pip install -r requirements.txt
python 01_generar_holograma.py
python 02_reconstruir_holograma.py
```

Si usas el entorno virtual del proyecto, activalo antes (`.venv\Scripts\Activate.ps1`)
o usa su interprete directamente:

```powershell
..\.venv\Scripts\python.exe 01_generar_holograma.py
..\.venv\Scripts\python.exe 02_reconstruir_holograma.py
```

En el primer programa, carga una imagen y ajusta `t0` y `beta` con los
deslizadores. Guarda el archivo `.npz`; luego abre el segundo programa y carga
ese archivo. El segundo programa permite volver a ajustar los mismos parametros
y guardar una copia con los valores nuevos. Longitud de onda, angulos, paso de
pixel y distancia de observacion quedan fijos para mantener solo los controles
solicitados.

El programa 2 tambien permite guardar la intensidad propagada actual como PNG
normalizado para visualizarla o como archivo `.npy` con los valores fisicos
originales sin normalizar.

## Modelo y unidades

La imagen en escala de grises se convierte en amplitud normalizada `A` y la fase
inicial es cero: `O=A exp(i phi)`. La onda de referencia es
`R=R0 exp(i(kx x + ky y + phi0))`, con `k=2 pi/lambda`,
`kx=k sin(theta_x)` y `ky=k sin(theta_y)`. Los dos angulos se interpretan como
componentes direccionales independientes, con `sin(theta_x)^2 + sin(theta_y)^2
<= 1`.

Las coordenadas, pasos de pixel y longitudes de onda se convierten a metros.
La irradiancia registrada es `|O+R|^2`; para mantener una escala estable en el
modelo lineal de pelicula, se divide por la cota `(1+R0)^2`. La transmision es
`t=t0+beta I_norm`. Los campos no se normalizan para visualizacion ni para
propagacion; las figuras solo ajustan el mapa de colores.

La iluminacion produce `U(x,y,0)=R_ill(x,y)t(x,y)`. La propagacion utiliza el
espectro angular con la funcion de transferencia
`H=exp(i z sqrt(k^2-kx^2-ky^2))`; las componentes evanescentes decaen y se
aplica un limite de banda para reducir el aliasing del plano de calculo.

El `.npz` conserva la transmision, la irradiancia normalizada, el campo del
objeto, la referencia de grabacion, la imagen y metadatos JSON (dimensiones,
paso espacial, longitud de onda, angulos y parametros de pelicula). Se valida
la version y la forma de los arrays al cargarlo.

## Pruebas

Desde esta carpeta, ejecuta:

```powershell
py -m unittest -v
```
