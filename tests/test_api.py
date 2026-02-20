def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Brain Tumor MRI Detection API" in response.json()["message"]


def test_predict_invalid_file(client):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "File must be an image only."


def test_predict_success(client):
    # The classifier is mocked in conftest.py, so we just need valid-looking input
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", b"fake image content", "image/jpeg")}
    )
    # Since we mocked the return value to be empty predictions
    assert response.status_code == 200
    assert "predictions" in response.json()
