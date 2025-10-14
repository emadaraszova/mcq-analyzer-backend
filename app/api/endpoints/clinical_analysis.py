from fastapi import APIRouter, HTTPException, Depends
from app.schemas.user_message import UserMessageAnalysis
from app.services.gemini_service import GeminiService
from app.services.openai_service import OpenAIService

router = APIRouter()

@router.post("/analyze-clinical", summary="Analyze Clinical Question")
async def analyze_clinical_question(user_message: UserMessageAnalysis):
    """
    Analyze a clinical question to extract structured information.
    """
    # Log the received model for debugging
    print(f"Model received: {user_message.model}")
    
    # Dynamically select the service based on the model
    if user_message.model == "gemini-2.5-flash":
        service = GeminiService()
    elif user_message.model == "gpt-4o":
        service = OpenAIService()
    else:
        # Log unsupported model case
        print(f"Unsupported model: {user_message.model}")
        raise HTTPException(
            status_code=400, detail=f"Unsupported model: {user_message.model}"
        )
    
    # Log message details
    print(f"Message: {user_message.message}")
    print(f"Number of questions: {user_message.number_of_questions}")

    # Call the service to extract structured information
    try:
        structured_data = service.extract_clinical_info(
            questions=user_message.message, 
            number_of_questions=user_message.number_of_questions
        )
        return {"structured_data": structured_data}
    except HTTPException as e:
        # Log HTTP-specific exceptions
        print(f"HTTPException: {e.detail}")
        raise e
    except Exception as e:
        # Log unexpected errors
        print(f"Unhandled error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
