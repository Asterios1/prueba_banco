
from pathlib import Path

class ETLConfig:
    """Clase para manejar la configuración del proceso ETL"""
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.insumos_dir = base_dir / "insumos"
        self.salidas_dir = base_dir / "salidas"
        self.logs_dir = base_dir / "logs"
        self.batch_size = 10000
        self.max_workers = 4

    def ensure_directories(self) -> None:
        """Asegura que existan todos los directorios necesarios"""
        for directory in [self.insumos_dir, self.salidas_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
