import kagglehub
import shutil
from pathlib import Path
import logging
from datetime import datetime
import sys
import json
import time
import traceback
from dataclasses import dataclass, field
from contextlib import contextmanager
import pandas as pd
import numpy as np
from typing import List, Dict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Tuple
import os

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
                error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

def setup_logging(config: ETLConfig) -> logging.Logger:
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

        # Handler para consola con colores
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


@contextmanager
def error_handler(monitor: ProcessMonitor, stage: str):
    try:
        yield
    except Exception as e:
        error_msg = f"Error en {stage}: {e}"
        monitor.update_status(stage, error=e, details=error_msg)
        logging.getLogger(__name__).error(f"{error_msg}\n{traceback.format_exc()}")
        raise

def clean_directories(monitor: ProcessMonitor) -> None:
    """
    Limpia la caché de kagglehub, la carpeta de insumos y la carpeta de salidas antes de iniciar el proceso.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Limpiar caché de kagglehub
        kaggle_cache = Path.home() / ".cache" / "kagglehub"
        monitor.update_status("limpiando_cache", 
                              details="Limpiando caché de kagglehub")
        
        if kaggle_cache.exists():
            logger.info(f"Limpiando caché de kagglehub en: {kaggle_cache}")
            shutil.rmtree(kaggle_cache, ignore_errors=True)
            logger.info("✓ Caché de kagglehub eliminada")
        else:
            logger.info("No se encontró caché de kagglehub para limpiar")
            
        # Limpiar carpeta de insumos
        base_dir = Path(__file__).parent
        insumos_dir = base_dir / "insumos"
        monitor.update_status("limpiando_insumos", 
                              details="Limpiando carpeta de insumos")
        
        if insumos_dir.exists():
            logger.info(f"Limpiando carpeta de insumos en: {insumos_dir}")
            for item in insumos_dir.glob("*"):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            logger.info("✓ Carpeta de insumos limpiada")
        
        # Limpiar carpeta de salidas
        salidas_dir = base_dir / "salidas"
        monitor.update_status("limpiando_salidas", 
                              details="Limpiando carpeta de salidas")
        
        if salidas_dir.exists():
            logger.info(f"Limpiando carpeta de salidas en: {salidas_dir}")
            for item in salidas_dir.glob("*"):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            logger.info("✓ Carpeta de salidas limpiada")
        
        monitor.update_status("limpieza_completada", 
                              details="Limpieza de directorios completada")
        
    except Exception as e:
        error_msg = f"Error durante la limpieza de directorios: {e}"
        logger.error(f"{error_msg}")
        monitor.update_status("error_limpieza", error=e, details=error_msg)
        raise RuntimeError(error_msg) from e


def setup_and_download_files(monitor: ProcessMonitor) -> None:
    logger = logging.getLogger(__name__)
    base_dir = Path(__file__).parent
    insumos_dir = base_dir / "insumos"
    
    try:
        # Create insumos directory if it doesn't exist
        insumos_dir.mkdir(exist_ok=True)
        logger.info(f"Verificando directorio de insumos: {insumos_dir}")
        
        # Download files
        monitor.update_status("descargando_archivos", 
                            details="Iniciando descarga de archivos desde Kaggle")
        logger.info("Iniciando descarga de archivos...")
        
        downloaded_paths = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews")
        logger.info(f"Archivos descargados en: {downloaded_paths}")
        
        # Move files to insumos directory
        monitor.update_status("moviendo_archivos", 
                            details="Moviendo archivos a directorio de insumos")
        
        downloaded_files = Path(downloaded_paths).glob("*")
        for file_path in downloaded_files:
            if file_path.is_file():
                dest_path = insumos_dir / file_path.name
                logger.info(f"Moviendo {file_path.name} a {dest_path}")
                
                shutil.move(str(file_path), str(dest_path))
                logger.info(f"✓ Archivo movido exitosamente: {file_path.name}")
                monitor.metrics.records_processed += 1
        
        monitor.update_status("archivos_preparados", 
                            details="Archivos descargados y movidos exitosamente")
        logger.info("✓ Todos los archivos han sido movidos al directorio de insumos")
        
    except Exception as e:
        error_msg = f"Error en la preparación de archivos: {e}"
        logger.error(f"{error_msg}")
        monitor.update_status("error_preparacion", error=e, details=error_msg)
        raise RuntimeError(error_msg) from e

class DataCleaner:
    """Clase para manejar la limpieza de datos"""
    
    def __init__(self, monitor: ProcessMonitor):
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)

    def clean_books_data(self, df: Optional[pd.DataFrame] = None, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Limpia y valida el DataFrame de libros"""
        if file_path:
            # Verificar si el archivo existe antes de cargarlo
            if not file_path.exists():
                self.logger.error(f"Archivo no encontrado: {file_path}")
                raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                self.logger.error(f"Error al leer el archivo {file_path}: {e}")
                raise e
        
        if df is None:
            raise ValueError("Se debe proporcionar un DataFrame o una ruta de archivo.")

        with self.monitor.track_time("clean_books"):
            df = df.copy()
            
            # Validar columnas requeridas
            required_columns = {'Title', 'authors', 'publishedDate'}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Columnas faltantes: {missing_columns}")

            # Limpiar datos
            df['Title'] = df['Title'].str.strip()
            df['authors'] = df['authors'].fillna('Desconocido')
            df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce')

            # Registrar métricas
            null_counts = df.isnull().sum()
            if null_counts.any():
                self.monitor.add_warning(f"Valores nulos encontrados: {null_counts[null_counts > 0].to_dict()}")

        return df

    def clean_ratings_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y valida el DataFrame de ratings"""
        with self.monitor.track_time("clean_ratings"):
            df = df.copy()
            
            # Validar columnas requeridas
            required_columns = {'Id', 'Title', 'review/score', 'Price'}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Columnas faltantes: {missing_columns}")

            # Limpiar datos
            df['review/score'] = pd.to_numeric(df['review/score'], errors='coerce')
            df = df.dropna(subset=['review/score'])
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)

            return df


class SentimentAnalyzer:
    """Clase mejorada para análisis de sentimientos"""
    def __init__(self, monitor: ProcessMonitor):
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
        
        # Asegurar recursos NLTK con manejo de errores mejorado
        with monitor.track_time("nltk_download"):
            try:
                # Descargar recursos necesarios
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                # Verificar que los recursos se descargaron correctamente
                if not nltk.data.find('tokenizers/punkt'):
                    raise LookupError("No se pudo encontrar el recurso 'punkt' después de la descarga")
                if not nltk.data.find('corpora/stopwords'):
                    raise LookupError("No se pudo encontrar el recurso 'stopwords' después de la descarga")
                
                self.stop_words = set(stopwords.words('english'))
                self.logger.info("Recursos NLTK descargados y verificados correctamente")
                
            except Exception as e:
                error_msg = f"Error crítico al inicializar recursos NLTK: {e}"
                self.logger.error(error_msg)
                self.monitor.add_warning(error_msg)
                # Establecer stopwords vacío como fallback
                self.stop_words = set()
                raise RuntimeError(error_msg) from e

    def process_batch(self, texts: List[str]) -> List[float]:
        """Procesa un lote de textos en paralelo"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(self.analyze_sentiment, texts))

    def analyze_sentiment(self, text: str) -> float:
        """Analiza el sentimiento de un texto individual"""
        try:
            if not isinstance(text, str):
                return 0.0
            
            processed_text = self._preprocess_text(text)
            if not processed_text:
                return 0.0
                
            return TextBlob(processed_text).sentiment.polarity
        except Exception as e:
            self.logger.error(f"Error en análisis de sentimientos: {e}")
            return 0.0

    def _preprocess_text(self, text: str) -> str:
        """Preprocesa el texto para análisis con manejo de errores mejorado"""
        try:
            # Verificar nuevamente la disponibilidad de punkt
            if not nltk.data.find('tokenizers/punkt'):
                self.logger.warning("Recurso 'punkt' no disponible, usando split() básico")
                tokens = text.lower().split()
            else:
                tokens = word_tokenize(text.lower())
            
            # Filtrar tokens
            cleaned_tokens = [word for word in tokens 
                            if word.isalpha() and 
                            (not self.stop_words or word not in self.stop_words)]
            
            return " ".join(cleaned_tokens) if cleaned_tokens else ""
            
        except Exception as e:
            self.logger.error(f"Error en preprocesamiento: {e}")
            # Retornar texto original como fallback
            return text.lower()

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str, 
                         batch_size: int = 1000) -> pd.Series:
        """Analiza sentimientos para todo un DataFrame en batches"""
        results = []
        total_rows = len(df)
        failed_analyses = 0
        
        with tqdm(total=total_rows, desc="Analizando sentimientos") as pbar:
            for i in range(0, total_rows, batch_size):
                try:
                    batch = df[text_column].iloc[i:i+batch_size].tolist()
                    batch_results = self.process_batch(batch)
                    results.extend(batch_results)
                    
                    # Contar análisis fallidos (sentiment = 0.0)
                    failed_analyses += sum(1 for r in batch_results if r == 0.0)
                    
                except Exception as e:
                    self.logger.error(f"Error en batch {i//batch_size}: {e}")
                    # Rellenar con valores neutros en caso de error
                    results.extend([0.0] * len(batch))
                    failed_analyses += len(batch)
                
                finally:
                    pbar.update(len(batch))
        
        # Registrar métricas de análisis
        failure_rate = (failed_analyses / total_rows) * 100
        if failure_rate > 10:
            self.monitor.add_warning(
                f"Alta tasa de fallos en análisis de sentimientos: {failure_rate:.1f}%"
            )
        
        return pd.Series(results, index=df.index)

class DataAnalyzer:
    """Clase para análisis de datos"""
    def __init__(self, monitor: ProcessMonitor):
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
        self.sentiment_analyzer = SentimentAnalyzer(monitor)

    def analyze_data(self, books_df: pd.DataFrame, ratings_df: pd.DataFrame) -> Dict[str, Any]:
        """Realiza análisis completo de los datos"""
        with self.monitor.track_time("complete_analysis"):
            try:
                results = {}
                
                # Análisis de valoraciones
                results.update(self._analyze_ratings(ratings_df))
                
                # Análisis de autores
                results.update(self._analyze_authors(books_df))
                
                # Análisis de sentimientos
                results.update(self._analyze_sentiments(ratings_df))
                
                return results
            
            except Exception as e:
                self.logger.error(f"Error en análisis de datos: {e}")
                raise

    def _analyze_ratings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza las valoraciones"""
        with self.monitor.track_time("ratings_analysis"):
            return {
                "avg_ratings": df.groupby('Title')['review/score'].agg(['mean', 'count']).to_dict(),
                "rating_distribution": df['review/score'].value_counts().sort_index().to_dict()
            }

    def _analyze_authors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza datos de autores"""
        with self.monitor.track_time("authors_analysis"):
            return {
                "top_authors": df['authors'].value_counts().head(10).to_dict(),
                "authors_books_count": df.groupby('authors').size().to_dict()
            }

    def _analyze_sentiments(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza sentimientos de las reseñas"""
        with self.monitor.track_time("sentiment_analysis"):
            # Calculate sentiments
            sentiments = self.sentiment_analyzer.analyze_dataframe(
                df, 'review/text', batch_size=1000
            )
            
            # Convert sentiments to a named series for easier handling
            sentiments.name = 'sentiment_score'
            
            # Calculate basic statistics
            sentiment_stats = sentiments.describe().to_dict()
            
            # Calculate average sentiment by rating
            sentiment_by_rating = df.assign(sentiment_score=sentiments).groupby('review/score')['sentiment_score'].mean().to_dict()
            
            # Bin sentiments into categories for distribution
            sentiment_bins = pd.cut(sentiments, 
                                  bins=[-1, -0.5, -0.1, 0.1, 0.5, 1], 
                                  labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
            sentiment_distribution = sentiment_bins.value_counts().to_dict()
            
            return {
                "sentiment_statistics": sentiment_stats,
                "sentiment_by_rating": sentiment_by_rating,
                "sentiment_distribution": sentiment_distribution
            }

def convert_to_serializable(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts complex data types to serializable formats
    
    Args:
        results: Dictionary of analysis results
    
    Returns:
        Serializable dictionary
    """
    serializable_results = {}
    
    for key, value in results.items():
        if isinstance(value, pd.Series):
            # Convert Series to dictionary
            serializable_results[key] = value.to_dict()
        elif isinstance(value, pd.DataFrame):
            # Convert DataFrame to dictionary
            serializable_results[key] = value.to_dict(orient='records')
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            serializable_results[key] = {
                k: (v.tolist() if isinstance(v, pd.Series) else 
                    v.to_dict() if isinstance(v, pd.DataFrame) else 
                    v) 
                for k, v in value.items()
            }
        else:
            serializable_results[key] = value
    
    return serializable_results

class BookRanker:
    """
    Clase para realizar ranking de libros según diferentes métricas
    """
    def __init__(self, monitor: ProcessMonitor):
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)

    def rank_by_review_count(self, ratings_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Rankea los libros por número total de reseñas
        
        Args:
            ratings_df: DataFrame de reseñas
            top_n: Número de libros a mostrar
        
        Returns:
            DataFrame con los libros más revisados
        """
        with self.monitor.track_time("rank_by_review_count"):
            review_counts = ratings_df['Title'].value_counts()
            top_books = review_counts.nlargest(top_n)
            return pd.DataFrame({
                'Título': top_books.index,
                'Número de Reseñas': top_books.values
            })

    def rank_by_average_score(self, ratings_df: pd.DataFrame, min_reviews: int = 5, top_n: int = 10) -> pd.DataFrame:
        """
        Rankea los libros por promedio de puntaje
        
        Args:
            ratings_df: DataFrame de reseñas
            min_reviews: Número mínimo de reseñas para considerar un libro
            top_n: Número de libros a mostrar
        
        Returns:
            DataFrame con los libros mejor puntuados
        """
        with self.monitor.track_time("rank_by_average_score"):
            book_ratings = ratings_df.groupby('Title')['review/score'].agg(['mean', 'count'])
            
            # Filtrar libros con número mínimo de reseñas
            filtered_books = book_ratings[book_ratings['count'] >= min_reviews]
            
            # Ordenar por promedio de puntaje
            top_books = filtered_books.sort_values('mean', ascending=False).head(top_n)
            
            return pd.DataFrame({
                'Título': top_books.index,
                'Puntaje Promedio': top_books['mean'].round(2),
                'Número de Reseñas': top_books['count']
            })

    def rank_by_sentiment_score(self, ratings_df: pd.DataFrame, sentiment_scores: pd.Series, 
                                 min_reviews: int = 5, top_n: int = 10) -> pd.DataFrame:
        """
        Rankea los libros por sentimiento promedio de las reseñas
        
        Args:
            ratings_df: DataFrame de reseñas
            sentiment_scores: Serie con puntajes de sentimiento
            min_reviews: Número mínimo de reseñas para considerar un libro
            top_n: Número de libros a mostrar
        
        Returns:
            DataFrame con los libros con mejor sentimiento
        """
        with self.monitor.track_time("rank_by_sentiment_score"):
            # Agregar puntajes de sentimiento al DataFrame
            book_sentiments = ratings_df.copy()
            book_sentiments['sentiment_score'] = sentiment_scores
            
            # Calcular métricas de sentimiento por libro
            sentiment_metrics = book_sentiments.groupby('Title').agg({
                'sentiment_score': ['mean', 'count']
            })
            
            # Filtrar libros con número mínimo de reseñas
            filtered_books = sentiment_metrics[sentiment_metrics[('sentiment_score', 'count')] >= min_reviews]
            
            # Ordenar por sentimiento promedio
            top_sentiment_books = filtered_books.sort_values(('sentiment_score', 'mean'), ascending=False).head(top_n)
            
            return pd.DataFrame({
                'Título': top_sentiment_books.index,
                'Sentimiento Promedio': top_sentiment_books[('sentiment_score', 'mean')].round(3),
                'Número de Reseñas': top_sentiment_books[('sentiment_score', 'count')]
            })

def export_results_to_txt(books_df: pd.DataFrame, ratings_df: pd.DataFrame, results: Dict[str, Any], output_path: Path) -> None:
    """
    Exporta los resultados del análisis a un archivo de texto formateado.
    
    Args:
        books_df: DataFrame con datos de libros
        ratings_df: DataFrame con datos de reseñas
        results: Diccionario con resultados del análisis
        output_path: Ruta donde se guardará el archivo
    """
    # Convertir resultados a tipos serializables
    results = convert_to_serializable(results)
    
    # Preparar ranker para los rankings
    monitor = ProcessMonitor()
    book_ranker = BookRanker(monitor)
    
    # Calcular puntajes de sentimiento
    sentiment_analyzer = SentimentAnalyzer(monitor)
    sentiment_scores = sentiment_analyzer.analyze_dataframe(ratings_df, 'review/text', batch_size=1000)
    
    # Generar rankings
    top_review_count = book_ranker.rank_by_review_count(ratings_df)
    top_average_score = book_ranker.rank_by_average_score(ratings_df)
    top_sentiment_score = book_ranker.rank_by_sentiment_score(ratings_df, sentiment_scores)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Escribir encabezado
        f.write("=== REPORTE DE ANÁLISIS DE LIBROS DE AMAZON ===\n")
        f.write(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Estadísticas generales
        f.write("=== ESTADÍSTICAS GENERALES ===\n")
        total_reviews = len(ratings_df)
        total_books = len(books_df)
        f.write(f"Total de reseñas analizadas: {total_reviews:,}\n")
        f.write(f"Total de libros únicos: {total_books:,}\n\n")
        
        # Valoraciones promedio por libro
        f.write("=== VALORACIONES PROMEDIO POR LIBRO ===\n")
        try:
            # Manejar diferentes estructuras de datos para avg_ratings
            if isinstance(results['avg_ratings'], dict):
                # Si es un diccionario anidado
                rated_books = [
                    (title, stats.get('mean', 0), stats.get('count', 0)) 
                    for title, stats in results['avg_ratings'].items()
                ]
            elif isinstance(results['avg_ratings'], list):
                # Si es una lista de diccionarios
                rated_books = [
                    (item.get('Title', 'Unknown'), 
                     item.get('mean', 0), 
                     item.get('count', 0)) 
                    for item in results['avg_ratings']
                ]
            else:
                # Fallback si la estructura no es la esperada
                rated_books = []
            
            # Ordenar y tomar top 10
            top_rated_books = sorted(rated_books, key=lambda x: x[1], reverse=True)[:10]
            
            for title, mean_rating, count in top_rated_books:
                f.write(f"{title[:50]:50} | Rating: {mean_rating:.2f} | Reseñas: {int(count):,}\n")
        
        except (KeyError, TypeError) as e:
            f.write(f"Error al procesar valoraciones: {str(e)}\n")
        f.write("\n")
        
        # Autores más populares
        f.write("=== AUTORES MÁS POPULARES ===\n")
        try:
            top_authors = results.get('top_authors', {})
            if isinstance(top_authors, dict):
                for author, count in sorted(top_authors.items(), key=lambda x: x[1], reverse=True)[:10]:
                    f.write(f"{author:50} | Libros: {int(count):,}\n")
            else:
                f.write("No se pudieron procesar los datos de autores.\n")
        except Exception as e:
            f.write(f"Error al procesar autores: {str(e)}\n")
        f.write("\n")
        
        # Análisis de sentimientos
        f.write("=== ANÁLISIS DE SENTIMIENTOS ===\n")
        try:
            sentiment_stats = results.get('sentiment_statistics', {})
            f.write(f"Sentimiento promedio general: {sentiment_stats.get('mean', 'N/A'):.3f}\n")
            f.write(f"Desviación estándar: {sentiment_stats.get('std', 'N/A'):.3f}\n\n")
        except Exception as e:
            f.write(f"Error al procesar estadísticas de sentimiento: {str(e)}\n")
        
        # Distribución de sentimientos
        f.write("=== DISTRIBUCIÓN DE SENTIMIENTOS ===\n")
        try:
            sentiment_dist = results.get('sentiment_distribution', {})
            total_sentiments = sum(sentiment_dist.values())
            for category, count in sentiment_dist.items():
                percentage = (count / total_sentiments) * 100 if total_sentiments > 0 else 0
                f.write(f"{category:15} | {count:,} reseñas ({percentage:.1f}%)\n")
        except Exception as e:
            f.write(f"Error al procesar distribución de sentimientos: {str(e)}\n")
        f.write("\n")
        
        # Sentimiento por valoración
        f.write("=== SENTIMIENTO PROMEDIO POR VALORACIÓN ===\n")
        try:
            sentiment_by_rating = results.get('sentiment_by_rating', {})
            for rating, sentiment in sorted(sentiment_by_rating.items()):
                f.write(f"Rating {rating}: {sentiment:.3f}\n")
        except Exception as e:
            f.write(f"Error al procesar sentimiento por valoración: {str(e)}\n")
        f.write("\n")
        
        # Libros con mejor sentimiento promedio
        f.write("=== TOP 10 LIBROS CON MEJOR SENTIMIENTO ===\n")
        try:
            book_sentiments = ratings_df.copy()
            book_sentiments['sentiment_score'] = book_sentiments['review/score'].map(results.get('sentiment_by_rating', {}))
            
            book_sentiment_summary = book_sentiments.groupby('Title').agg({
                'sentiment_score': 'mean',
                'review/score': 'count'
            }).rename(columns={'review/score': 'count'})
            
            # Filtrar libros con al menos 5 reseñas
            book_sentiment_summary = book_sentiment_summary[book_sentiment_summary['count'] >= 5]
            top_sentiment_books = book_sentiment_summary.sort_values('sentiment_score', ascending=False).head(10)
            
            for title, row in top_sentiment_books.iterrows():
                f.write(f"{title[:50]:50} | Sentimiento: {row['sentiment_score']:.3f} | Reseñas: {int(row['count']):,}\n")
        except Exception as e:
            f.write(f"Error al procesar libros con mejor sentimiento: {str(e)}\n")
        f.write("\n")
        
        # Rankings adicionales de libros
        f.write("\n=== RANKINGS ADICIONALES DE LIBROS ===\n\n")
        
        # Top libros por número de reseñas
        f.write("=== TOP 10 LIBROS POR NÚMERO DE RESEÑAS ===\n")
        for idx, row in top_review_count.iterrows():
            f.write(f"{row['Título'][:50]:50} | Reseñas: {int(row['Número de Reseñas']):,}\n")
        f.write("\n")
        
        # Top libros por puntaje promedio
        f.write("=== TOP 10 LIBROS POR PUNTAJE PROMEDIO ===\n")
        for idx, row in top_average_score.iterrows():
            f.write(f"{row['Título'][:50]:50} | Puntaje: {row['Puntaje Promedio']:.2f} | Reseñas: {int(row['Número de Reseñas']):,}\n")
        f.write("\n")
        
        # Top libros por sentimiento promedio
        f.write("=== TOP 10 LIBROS POR SENTIMIENTO PROMEDIO ===\n")
        for idx, row in top_sentiment_score.iterrows():
            f.write(f"{row['Título'][:50]:50} | Sentimiento: {row['Sentimiento Promedio']:.3f} | Reseñas: {int(row['Número de Reseñas']):,}\n")
            
def main():
    """Función principal mejorada"""
    try:
        # Inicializar configuración
        config = ETLConfig(Path(__file__).parent)
        config.ensure_directories()
        
        # Inicializar logging y monitor
        logger = setup_logging(config)
        monitor = ProcessMonitor()
        
        logger.info("=== Iniciando proceso ETL ===")
        
        # Inicializar componentes
        data_cleaner = DataCleaner(monitor)
        data_analyzer = DataAnalyzer(monitor)
        
        # Proceso principal
        with monitor.track_time("total_process"):
            # Limpiar directorios y descargar datos
            clean_directories(monitor)
            setup_and_download_files(monitor)
            
            # Cargar y limpiar datos
            books_df = data_cleaner.clean_books_data(
                pd.read_csv(config.insumos_dir / "books_data.csv")
            )
            ratings_df = data_cleaner.clean_ratings_data(
                pd.read_csv(config.insumos_dir / "Books_rating.csv")
            )
            
            # Realizar análisis
            results = data_analyzer.analyze_data(books_df, ratings_df)
            
            # Convertir resultados a tipos serializables
            serializable_results = convert_to_serializable(results)
            
            # Guardar resultados en JSON
            results_file = config.salidas_dir / "analysis_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            # Exportar resultados en formato de texto
            txt_results_file = config.salidas_dir / "analysis_report.txt"
            export_results_to_txt(books_df, ratings_df, results, txt_results_file)
        
        logger.info("=== Proceso completado exitosamente ===")
        logger.info(f"Reporte generado en: {txt_results_file}")
        
    except Exception as e:
        logger.error(f"Error crítico en el proceso: {e}")
        # Imprimir traza completa para diagnóstico
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
