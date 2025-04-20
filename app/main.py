from fastapi import FastAPI
from .api import router

app = FastAPI(
    title="Candidate Selection API",
    description="API for predicting candidate hiring status using SVM",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 