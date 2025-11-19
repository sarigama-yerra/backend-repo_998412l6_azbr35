"""
Database Schemas for Smriti (AI-Powered Flashcards)

Each Pydantic model corresponds to a MongoDB collection. Collection name is the lowercase of the class name.
"""
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime


class User(BaseModel):
    email: EmailStr
    password_hash: str
    createdAt: datetime = Field(default_factory=datetime.utcnow)


class Deck(BaseModel):
    userId: str
    name: str
    description: Optional[str] = None
    totalCards: int = 0
    studyTime: int = 0  # seconds accumulated
    lastStudied: Optional[datetime] = None


class FlashCard(BaseModel):
    deckId: str
    userId: str
    question: str
    answer: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    lastReviewed: Optional[datetime] = None
    reviewCount: int = 0


class Settings(BaseModel):
    userId: str
    geminiApiKey: Optional[str] = None
    dailyGoal: int = 20
    theme: Literal["light", "dark", "system"] = "light"


# Public schema endpoint consumer expects these to exist; do not rename classes.
