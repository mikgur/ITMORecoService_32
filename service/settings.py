from pydantic import BaseSettings


class Config(BaseSettings):

    class Config:
        case_sensitive = False


class LogConfig(Config):
    level: str = "INFO"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"

    class Config:
        case_sensitive = False
        fields = {
            "level": {
                "env": ["log_level"]
            },
        }


class ServiceConfig(Config):
    service_name: str = "reco_service"
    k_recs: int = 10
    models_dir: str = "models"

    offline_tfidf_idf_10: str = "tfidf_idf_10_offline.pkl"
    offline_mf_ann: str = "offline_ann_recs.pkl"
    online_tfidf_idf_10: str = "TFIDF_10.dill"

    postprocessing: str = 'PopularPadding'
    popular_name: str = "popular.pkl"

    train_name: str = "train.csv"

    log_config: LogConfig


def get_config() -> ServiceConfig:
    return ServiceConfig(
        log_config=LogConfig(),
    )
