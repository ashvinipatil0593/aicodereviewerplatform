import json

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ai_engine import ai_review
from database import Base, SessionLocal, engine
from models import CodeReview

Base.metadata.create_all(bind=engine)

app = FastAPI(title="DevAudit AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    language: str
    code: str


@app.get("/")
def home():
    return {"message": "Backend is running"}


def save_review_to_db(code: str, language: str, result: dict):
    db: Session = SessionLocal()

    try:
        row = CodeReview(
            code=code,
            language=language,
            errors=json.dumps(result.get("errors", [])),
            suggestions=json.dumps(result.get("suggestions", [])),
            output=result.get("output", ""),
            fixed_code=result.get("fixedCode", ""),
            score=json.dumps(result.get("score", {})),
            dl_insights=json.dumps(result.get("dlInsights", {})),
        )

        db.add(row)
        db.commit()

    except Exception as db_error:
        print("DB Error:", db_error)

    finally:
        db.close()


@app.post("/review")
def review_code(request: ReviewRequest, background_tasks: BackgroundTasks):
    try:
        result = ai_review(request.language, request.code)

    except Exception as e:
        result = {
            "errors": [
                {
                    "line": 0,
                    "message": f"AI Engine Error: {str(e)}",
                    "fix": "Check model / backend logs",
                }
            ],
            "suggestions": [],
            "output": "Error while processing code",
            "fixedCode": request.code,
            "score": {
                "readability": 0,
                "performance": 0,
                "maintainability": 0,
                "security": 0,
            },
            "dlInsights": {
                "trainedExamples": 0,
                "similarExamplesFound": 0,
                "confidence": 0,
                "learnedSuggestions": [],
                "status": "AI failed",
            },
        }

    background_tasks.add_task(
        save_review_to_db,
        request.code,
        request.language,
        result,
    )

    return result