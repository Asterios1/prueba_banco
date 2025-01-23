
import time
import json
from pathlib import Path
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from contextlib import contextmanager
import traceback

@dataclass
class ProcessMetrics:
    start_time: float
    records_processed: int = 0
    records_failed: int = 0
    current_stage: str = "iniciando"
    error_details: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "Duracion": round(time.time() - self.start_time, 2),
            "Registros procesados": self.records_processed,
            "Registros fallidos": self.records_failed,
            "Proceso en ejecucion": self.current_stage,
            "Detalles de error": self.error_details,
            "Advertencias": self.warnings,
            "Metricas de rendimiento": self.performance_metrics
        }

class ProcessMonitor:
    def __init__(self, status_file: Path = Path("process_status.json")):
        self.metrics = ProcessMetrics(start_time=time.time())
        self.status_file = status_file
        self.logger = logging.getLogger(__name__)

    def add_warning(self, warning: str) -> None:
        self.metrics.warnings.append(warning)
        self.logger.warning(warning)

    def add_performance_metric(self, name: str, value: float) -> None:
        """Añade una métrica de rendimiento al monitor"""
        self.metrics.performance_metrics[name] = value
        self.logger.info(f"Métrica de rendimiento - {name}: {value}")

    @contextmanager
    def track_time(self, operation_name: str):
        """Contexto para medir el tiempo de operaciones"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.add_performance_metric(f"{operation_name}_duration", duration)

    def update_status(self, stage: str, records_processed: Optional[int] = None, 
                     error: Optional[Exception] = None, details: str = "") -> None:
        try:
            previous_stage = self.metrics.current_stage
            self.metrics.current_stage = stage
            
            self.logger.info(f"Cambiando etapa: {previous_stage} → {stage}")
            if details:
                self.logger.info(f"Detalles: {details}")

            if records_processed is not None:
                self.metrics.records_processed = records_processed
                self.logger.info(f"Registros procesados: {records_processed}")

            if error:
                self.metrics.records_failed += 1
                error_time = time.strftime("%Y-%m-%d %H:%M:%S")
                error_details = {
                    "mensaje": str(error),
                    "tipo": type(error).__name__,
                    "traceback": traceback.format_exc()
                }
                self.metrics.error_details[error_time] = error_details
                self.logger.error(f"Error en {stage}: {error}\n{traceback.format_exc()}")

            self._save_status()
        except Exception as e:
            self.logger.error(f"Error al actualizar estado: {e}\n{traceback.format_exc()}")

    def _save_status(self) -> None:
        """Método privado para guardar el estado actual"""
        try:
            self.status_file.write_text(
                json.dumps(self.metrics.to_dict(), indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
        except Exception as e:
            self.logger.error(f"Error al guardar estado: {e}")
