import os

import pytest


@pytest.fixture
def database_url():
    value = os.environ.get("TNSIM_TEST_DATABASE_URL")
    if not value:
        pytest.skip("TNSIM_TEST_DATABASE_URL is required for PostgreSQL integration")
    return value
