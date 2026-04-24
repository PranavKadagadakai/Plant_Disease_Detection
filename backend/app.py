from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import evaluate, predict, train

app = FastAPI(title="Plant Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(train.router)
app.include_router(evaluate.router)
app.include_router(predict.router)


@app.get("/")
def root():
    return {"message": "API running"}
