"""Database configuration for TNSIM."""

import os
from typing import Optional
from urllib.parse import quote


class DatabaseConfig:
    """PostgreSQL database connection configuration."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        self.host = os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", "localhost"))
        self.port = int(os.getenv("POSTGRES_PORT", os.getenv("DB_PORT", "5432")))
        self.database = os.getenv("POSTGRES_DB", os.getenv("DB_NAME", "tnsim"))
        self.username = os.getenv("POSTGRES_USER", os.getenv("DB_USER", "tnsim_user"))
        self.password = os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD", ""))
        self.min_connections = int(os.getenv("DB_MIN_CONNECTIONS", "5"))
        self.max_connections = int(os.getenv("DB_MAX_CONNECTIONS", "20"))
        self.connection_timeout = int(os.getenv("DB_CONNECTION_TIMEOUT", "30"))
        self.command_timeout = int(os.getenv("DB_COMMAND_TIMEOUT", "60"))
        self.ssl_mode = os.getenv("DB_SSL_MODE", "prefer")

    @property
    def dsn(self) -> str:
        """Get DSN string for PostgreSQL connection."""
        if os.getenv("DATABASE_URL"):
            return os.environ["DATABASE_URL"]
        return self.asyncpg_dsn + f"?sslmode={quote(self.ssl_mode, safe='')}"

    @property
    def asyncpg_dsn(self) -> str:
        """Get a URL-escaped DSN, giving DATABASE_URL precedence."""
        if os.getenv("DATABASE_URL"):
            return os.environ["DATABASE_URL"]
        host = f"[{self.host}]" if ":" in self.host else self.host
        return (
            f"postgresql://{quote(self.username, safe='')}:{quote(self.password, safe='')}"
            f"@{host}:{self.port}/{quote(self.database, safe='')}"
        )

    def get_connection_params(self) -> dict:
        """Get connection parameters for asyncpg."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.username,
            "password": self.password,
            "min_size": self.min_connections,
            "max_size": self.max_connections,
            "command_timeout": self.command_timeout,
            "server_settings": {"application_name": "tnsim_api", "timezone": "UTC"},
        }

    def validate(self) -> bool:
        """Validate configuration correctness."""
        required_fields = [self.host, self.database, self.username]
        return all(field for field in required_fields)

    def __repr__(self) -> str:
        """String representation of configuration (without password)."""
        return (
            f"DatabaseConfig(host='{self.host}', port={self.port}, "
            f"database='{self.database}', username='{self.username}', "
            f"min_connections={self.min_connections}, max_connections={self.max_connections})"
        )


# Global configuration instance
db_config = DatabaseConfig()


# Settings for different environments
class EnvironmentConfig:
    """Configurations for different environments."""

    @staticmethod
    def development() -> DatabaseConfig:
        """Configuration for development."""
        config = DatabaseConfig()
        config.database = os.getenv("POSTGRES_DB", os.getenv("DB_NAME", "tnsim_dev"))
        config.min_connections = 2
        config.max_connections = 10
        return config

    @staticmethod
    def testing() -> DatabaseConfig:
        """Configuration for testing."""
        config = DatabaseConfig()
        config.database = os.getenv("POSTGRES_DB", os.getenv("DB_NAME", "tnsim_test"))
        config.min_connections = 1
        config.max_connections = 5
        return config

    @staticmethod
    def production() -> DatabaseConfig:
        """Configuration for production."""
        config = DatabaseConfig()
        # In production, all parameters should be in environment variables
        if not config.validate():
            raise ValueError(
                "Not all required DB parameters are configured for production"
            )
        return config


def get_config(environment: Optional[str] = None) -> DatabaseConfig:
    """Get configuration for the specified environment."""
    env = environment or os.getenv("ENVIRONMENT", "development")

    if env == "development":
        return EnvironmentConfig.development()
    elif env == "testing":
        return EnvironmentConfig.testing()
    elif env == "production":
        return EnvironmentConfig.production()
    else:
        return db_config


# Constants for migrations
MIGRATIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "migrations")
MIGRATIONS_TABLE = "schema_migrations"


class MigrationConfig:
    """Configuration for database migrations."""

    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self.migrations_dir = MIGRATIONS_DIR
        self.migrations_table = MIGRATIONS_TABLE

    def get_migration_files(self) -> list[str]:
        """Get list of migration files."""
        import glob

        pattern = os.path.join(self.migrations_dir, "*.sql")
        files = glob.glob(pattern)
        return sorted(files)

    def get_migration_name(self, filepath: str) -> str:
        """Get migration name from file path."""
        return os.path.basename(filepath).replace(".sql", "")
