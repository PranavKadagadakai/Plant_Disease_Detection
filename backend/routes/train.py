from fastapi import APIRouter

from services.train_service import train_model

router = APIRouter()


@router.post("/train")
def train():
    return train_model()
