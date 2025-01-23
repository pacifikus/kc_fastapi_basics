from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Best regression model"}


def test_get_item_path():
    item_id = 1
    response = client.get(f"/path/{item_id}")
    assert response.status_code == 200
    assert response.json() == {"item_id": f"{item_id}"}


def test_predict():
    response = client.post(
        "/predict",
        headers={"Content-type": "application/json"},
        json={
              "age": 0,
              "sex": 0,
              "bmi": 0,
              "bp": 0,
              "s1": 0,
              "s2": 0,
              "s3": 0,
              "s4": 0,
              "s5": 0,
              "s6": 0
        },
    )
    assert response.status_code == 200
    assert response.json() == {"result": 153.52592923689}
