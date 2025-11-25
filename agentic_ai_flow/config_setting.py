from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class ConfigSetting(BaseSettings):
    """
    A class to represent configuration settings for an application.
    """

    # Data paths
    base_path: Path = Path(__file__).parent
    test_data_path: Path = base_path / "data" / "test_lite.pkl"
    train_data_path: Path = base_path / "data" / "train_lite.pkl"
    ensemble_data_path: Path = base_path / "data" / "ensemble_data.pkl"

    # DB paths
    chromadb_path: Path = base_path / "db" / "products_vectorstore"
    collection_name: str = "products"

    # model paths
    llm_base_model_name: str = "meta-llama/Llama-2-7b-hf"
    victorization_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    linear_regression_model_path: Path = (
        base_path / "training" / "ml_model" / "linear_regression_ensemble_model.pkl"
    )
    random_forest_model_path: Path = (
        base_path / "training" / "ml_model" / "random_forest_model.pkl"
    )
