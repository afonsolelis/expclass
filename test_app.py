import os
import pytest
from app import app

os.environ["VALID_TOKEN"] = "123456"

VALID_TOKEN = os.getenv("VALID_TOKEN")

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_unauthorized_train_endpoint(client):
    response = client.get('/train')
    assert response.status_code == 401

def test_train_endpoint(client):
    response = client.get('/train?token=123456')
    assert response.status_code == 200
    data = response.get_json()

    # # Verificar as chaves no JSON retornado
    # assert "accuracy" in data
    # assert "setosa" in data

    # # Verificar se a acurácia é um valor numérico
    # assert isinstance(data["accuracy"], float)
    # assert 0 <= data["accuracy"] <= 1

# 1. Ajustar o teste
# 2. Criar o actions
# 3. Configurar o MLOPS