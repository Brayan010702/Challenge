import fastapi

app = fastapi.FastAPI()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "model_loaded",
    }


@app.post("/predict", status_code=200)
async def post_predict() -> dict:
    return
