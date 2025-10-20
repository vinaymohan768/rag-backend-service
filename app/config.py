from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ragdb"
    postgres_user: str = "rag"
    postgres_password: str = "rag"

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_model: str = "gpt-4o-mini"

    default_chunk_strategy: str = "sentence"
    default_chunk_size: int = 512
    default_chunk_overlap: int = 64

    default_top_k: int = 8
    default_rerank_top_k: int = 3
    hybrid_alpha: float = 0.7  # 1.0 = pure vector, 0.0 = pure BM25

    @property
    def db_dsn(self) -> str:
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"dbname={self.postgres_db} user={self.postgres_user} "
            f"password={self.postgres_password}"
        )

    model_config = {"env_file": ".env"}


settings = Settings()
