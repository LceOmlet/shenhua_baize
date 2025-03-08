from pydantic import BaseModel

class ExtractionResult(BaseModel):
    content_type: str  # image/audio/text
    original_data: str
    extracted_fields: dict
    confidence: float
