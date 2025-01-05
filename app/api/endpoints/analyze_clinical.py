from fastapi import APIRouter, HTTPException
from app.schemas.user_message import UserMessage
from app.schemas.user_message import UserMessageAnalysis
from app.services.gemini_service import GeminiService
from app.services.openai_service import OpenAIService

router = APIRouter()

@router.post("/analyze-clinical", summary="Analyze Clinical Question")
async def analyze_clinical_question(user_message: UserMessageAnalysis):
    """
    Analyze a clinical question to extract structured information.
    """
    print(user_message.model)
    try:
        if user_message.model == "gemini-1.5-flash":
            service = GeminiService()
        elif user_message.model == "gpt-4o":
            service = OpenAIService()
        else:
            
            print("som tu")
            raise HTTPException(
                status_code=400, detail=f"Unsupported model: {user_message.model}"
            )
        
        print("message:", user_message.message)
        # Call the Gemini service to extract information
        structured_data = service.extract_clinical_info(questions=user_message.message, number_of_questions=user_message.number_of_questions)
        print("number of questions:", user_message.number_of_questions)
        return {"structured_data": structured_data}
    except HTTPException as e:
        print(f"HTTPException: {e.detail}")
        raise e
    except Exception as e:
        print(f"Error while analyzing clinical question: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
