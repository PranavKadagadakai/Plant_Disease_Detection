from fastapi import APIRouter, File, UploadFile

from services.detection_service import predict

router = APIRouter()


@router.post("/detect")
async def detection_route(file: UploadFile = File(...)):
    result = await predict(file)
    return result
