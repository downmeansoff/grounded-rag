from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "grounded_rag"
    postgres_user: str = "grounded_rag"
    postgres_password: str = "grounded_rag"

    # Бэкенд эмбеддингов: local (sentence-transformers, офлайн и бесплатно)
    # или gigachat (сеть и платный тариф). Размерность у моделей разная, а
    # колонка в базе заводится под конкретное число, поэтому смена бэкенда
    # означает пересбор индекса, а не только правку этих двух строк.
    embedding_backend: str = "local"
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768

    # Rerank выключен по умолчанию: он тянет вторую модель в память рядом
    # с эмбеддером и заметно замедляет ответ. Включать осознанно.
    use_rerank: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_candidates: int = 30

    # Contextual Retrieval: вызов LLM на каждый чанк при ingest. Выключен по
    # умолчанию, потому что это единственная часть индексации, которая тратит
    # платный ресурс. Результат кэшируется на диск, повторный ingest бесплатный.
    use_contextual: bool = False
    contextual_cache_path: str = ".cache/contexts.json"
    contextual_head_chars: int = 1200

    gigachat_credentials: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_model: str = "GigaChat-2"
    gigachat_embedding_model: str = "Embeddings"
    gigachat_embedding_batch: int = 32

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
