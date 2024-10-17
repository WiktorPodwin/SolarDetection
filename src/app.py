from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router

app = FastAPI()

# Include the API routes
app.include_router(api_router, prefix="/api")

# Add the Earth Engine API
# ee_engine = EarthEngineEngine()
# app.state.ee_engine = ee_engine
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

# mount static folder in the app
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load html templates
templates = Jinja2Templates(directory="templates")


# display map
@app.get("/")
async def ee_map(request: Request):
    return templates.TemplateResponse("map.html", {"request": request})
