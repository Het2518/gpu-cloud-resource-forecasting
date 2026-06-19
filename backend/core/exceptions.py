from fastapi import HTTPException

class ModelInferenceError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=detail)
