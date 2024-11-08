from typing import List
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class BaseResponse(BaseModel):
    response_code: int
    status: str


class BaseRouter:
    def __init__(self) -> None:
        pass

    def get_router(self) -> APIRouter:
        """Creates and returns an APIRouter with all the route definitions."""
        router = APIRouter()

        @router.get("/", response_model=BaseResponse)
        async def read_root():
            return {
                "response_code": 200,
                "status": "Welcome to the Solar Detection API!",
            }

        return router
