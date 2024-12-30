from fastapi import APIRouter, HTTPException
from app.schemas.user_message import UserMessage
from app.services.gemini_service import GeminiService

router = APIRouter()

@router.post("/analyze-clinical", summary="Analyze Clinical Question")
async def analyze_clinical_question(user_message: UserMessage):
    """
    Analyze a clinical question to extract structured information.
    """
    try:
        service = GeminiService()
        # Call the Gemini service to extract information
        structured_data = service.extract_clinical_info(user_message.message)
        print("result:", structured_data)
        return {"structured_data": structured_data}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error while analyzing clinical question: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
