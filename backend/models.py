from sqlalchemy import Column, Integer, String, Text
from database import Base


class CodeReview(Base):
    __tablename__ = "code_reviews"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(Text, nullable=False)
    language = Column(String, nullable=False)

    errors = Column(Text)          # JSON string
    suggestions = Column(Text)     # JSON string
    output = Column(Text)
    fixed_code = Column(Text)
    score = Column(Text)           # JSON string
    dl_insights = Column(Text)     # JSON string