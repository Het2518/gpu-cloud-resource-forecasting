from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/predict")
def predict():
    # TODO: Implement inference endpoint
    return {"prediction": "dummy"}
