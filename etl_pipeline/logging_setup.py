
import logging
from datetime import datetime
from pathlib import Path

def setup_logging(config) -> logging.Logger:
    """Configuración mejorada del sistema de logging"""
    try:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Formato detallado para los logs
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)

        # Handler para archivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            config.logs_dir / f"etl_process_{timestamp}.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info("Sistema de logging inicializado correctamente")
        return logger
    except Exception as e:
        print(f"Error crítico al configurar logging: {e}")
        raise
