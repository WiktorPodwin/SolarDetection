from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router

app = FastAPI()

# Include the API routes
app.include_router(api_router)

# Add the Earth Engine API
#
# # Add the ML model
# ml_model = MLModel()
# app.state.ml_model = ml_model

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def ee_map(request: Request):
    return JSONResponse(content={"message": "Welcome to the Solar Detection API!"})
