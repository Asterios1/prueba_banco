
from etl_pipeline.config import ETLConfig
from etl_pipeline.logging_setup import setup_logging
from etl_pipeline.process_monitor import ProcessMonitor

def main():
    config = ETLConfig(base_dir=Path(__file__).parent)
    config.ensure_directories()
    
    logger = setup_logging(config)
    monitor = ProcessMonitor()
    
    logger.info("Sistema modular iniciado correctamente.")

if __name__ == "__main__":
    main()
