from fastapi import APIRouter

from services.eval_service import evaluate_model

router = APIRouter()


@router.get("/evaluate")
def evaluate():
    return evaluate_model()
