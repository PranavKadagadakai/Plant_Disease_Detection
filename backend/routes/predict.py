from fastapi import APIRouter, File, UploadFile

from services.predict_service import predict

router = APIRouter()


@router.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    result = await predict(file)
    return result
