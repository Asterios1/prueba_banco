
# ETL Pipeline Project

Este proyecto implementa un sistema modular de ETL (Extract, Transform, Load) para analizar datos relacionados con libros y reseñas, utilizando Python y prácticas modernas de desarrollo.

## Propósito del Proyecto

El objetivo de este proyecto es proporcionar una solución robusta para limpiar, analizar y generar informes a partir de grandes volúmenes de datos. Se enfoca en:
- Descargar y organizar datos automáticamente.
- Limpiar y validar datos de libros y reseñas.
- Realizar análisis de datos, como valoraciones promedio y análisis de sentimientos.
- Generar informes detallados en formato JSON y texto.

## Estructura del Proyecto

```plaintext
etl_project/
├── etl_pipeline/
│   ├── __init__.py
│   ├── config.py
│   ├── logging_setup.py
│   ├── process_monitor.py
├── main.py
├── README.md
├── requirements.txt
└── insumos/  # Archivos de entrada (se crean automáticamente)
└── salidas/  # Archivos de salida (se crean automáticamente)
└── logs/     # Archivos de log (se crean automáticamente)
```

## Configuración del Entorno

### Requisitos Previos
- Python 3.8 o superior
- `pip` para la gestión de paquetes
- (Opcional) Un entorno virtual como `venv` o `conda`

### Instalación
1. Clona este repositorio o descomprime el archivo del proyecto.
2. Navega al directorio raíz del proyecto:
   ```bash
   cd etl_project
   ```
3. Crea un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
4. Instala las dependencias del proyecto:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución del Proyecto

1. Asegúrate de haber configurado el entorno.
2. Ejecuta el archivo `main.py`:
   ```bash
   python main.py
   ```
3. Durante la ejecución, se crearán automáticamente las carpetas `insumos`, `salidas` y `logs`.
   - Los datos procesados se guardarán en `salidas/`.
   - Los registros de la ejecución estarán disponibles en `logs/`.

## Notas Adicionales

- El archivo principal (`main.py`) utiliza las clases del paquete `etl_pipeline` para manejar configuraciones, monitoreo y logging.
- Puedes personalizar la configuración editando el archivo `config.py`.


