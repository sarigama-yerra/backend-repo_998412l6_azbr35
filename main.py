import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

from database import db, create_document, get_documents
from bson import ObjectId

# Optional: Google Gemini
try:
    import google.generativeai as genai
except Exception:  # package may not be installed yet
    genai = None

# Environment / JWT settings
SECRET_KEY = os.getenv("JWT_SECRET", "supersecret-smriti-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

app = FastAPI(title="Smriti API")

# CORS: allow any origin, no credentials (to comply with wildcard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------- Utility helpers -----------------------
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserPublic(BaseModel):
    id: str
    email: EmailStr
    createdAt: datetime


class DeckCreate(BaseModel):
    name: str
    description: Optional[str] = None


class DeckUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class CardCreate(BaseModel):
    question: str
    answer: str
    difficulty: Optional[str] = "medium"


class CardUpdate(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    difficulty: Optional[str] = None


class GenerateTopicRequest(BaseModel):
    topic: str
    deckName: Optional[str] = None
    num_cards: int = 12


class SettingsUpdate(BaseModel):
    geminiApiKey: Optional[str] = None
    dailyGoal: Optional[int] = None
    theme: Optional[str] = None


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def _id(obj: Any) -> str:
    return str(obj.get("_id")) if isinstance(obj, dict) and obj.get("_id") else str(obj)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db["user"].find_one({"_id": ObjectId(user_id)})
    if not user:
        raise credentials_exception
    return user


# ----------------------- Basic routes -----------------------
@app.get("/")
def root():
    return {"message": "Smriti API is running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Connected"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()[:10]
    except Exception as e:
        response["database"] = f"⚠️ {str(e)[:80]}"
    return response


# ----------------------- Auth -----------------------
@app.post("/api/auth/signup", response_model=UserPublic)
def signup(payload: UserCreate):
    existing = db["user"].find_one({"email": payload.email.lower()})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_doc = {
        "email": payload.email.lower(),
        "password_hash": get_password_hash(payload.password),
        "createdAt": datetime.utcnow(),
    }
    inserted = db["user"].insert_one(user_doc)
    # Also create default settings
    db["settings"].insert_one({
        "userId": str(inserted.inserted_id),
        "geminiApiKey": None,
        "dailyGoal": 20,
        "theme": "light",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    })
    return {"id": str(inserted.inserted_id), "email": user_doc["email"], "createdAt": user_doc["createdAt"]}


@app.post("/api/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = db["user"].find_one({"email": form_data.username.lower()})
    if not user or not verify_password(form_data.password, user.get("password_hash", "")):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    access_token = create_access_token({"sub": str(user["_id"])})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/auth/me", response_model=UserPublic)
async def me(current_user: dict = Depends(get_current_user)):
    return {"id": str(current_user["_id"]), "email": current_user["email"], "createdAt": current_user["createdAt"]}


# ----------------------- Settings -----------------------
@app.get("/api/settings")
async def get_settings(current_user: dict = Depends(get_current_user)):
    s = db["settings"].find_one({"userId": str(current_user["_id"])})
    if not s:
        s = {"userId": str(current_user["_id"]), "dailyGoal": 20, "theme": "light", "geminiApiKey": None}
        db["settings"].insert_one(s)
    # Never return geminiApiKey directly; only return whether set
    return {
        "dailyGoal": s.get("dailyGoal", 20),
        "theme": s.get("theme", "light"),
        "hasGeminiKey": bool(s.get("geminiApiKey")),
    }


@app.put("/api/settings")
async def update_settings(update: SettingsUpdate, current_user: dict = Depends(get_current_user)):
    set_ops: Dict[str, Any] = {}
    if update.dailyGoal is not None:
        set_ops["dailyGoal"] = max(1, int(update.dailyGoal))
    if update.theme is not None:
        if update.theme not in ("light", "dark", "system"):
            raise HTTPException(status_code=400, detail="Invalid theme")
        set_ops["theme"] = update.theme
    if update.geminiApiKey is not None:
        set_ops["geminiApiKey"] = update.geminiApiKey.strip() or None
    if not set_ops:
        return {"updated": False}
    set_ops["updated_at"] = datetime.utcnow()
    db["settings"].update_one({"userId": str(current_user["_id"])}, {"$set": set_ops}, upsert=True)
    return {"updated": True}


# ----------------------- Decks & Cards -----------------------
@app.get("/api/dashboard/stats")
async def dashboard_stats(current_user: dict = Depends(get_current_user)):
    uid = str(current_user["_id"])
    decks = list(db["deck"].find({"userId": uid}))
    total_decks = len(decks)
    deck_ids = [str(d["_id"]) for d in decks]
    total_cards = db["flashcard"].count_documents({"userId": uid})
    study_time = sum(int(d.get("studyTime", 0)) for d in decks)
    # Recently studied decks
    recents = list(db["deck"].find({"userId": uid, "lastStudied": {"$ne": None}}).sort("lastStudied", -1).limit(5))
    recents_fmt = [{"id": str(d["_id"]), "name": d.get("name"), "lastStudied": d.get("lastStudied") } for d in recents]
    settings = db["settings"].find_one({"userId": uid}) or {"dailyGoal": 20}
    return {
        "totalDecks": total_decks,
        "totalCards": total_cards,
        "totalStudyTime": study_time,
        "recentDecks": recents_fmt,
        "dailyGoal": settings.get("dailyGoal", 20),
    }


@app.get("/api/decks")
async def list_decks(q: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    uid = str(current_user["_id"])
    flt: Dict[str, Any] = {"userId": uid}
    if q:
        flt["name"] = {"$regex": q, "$options": "i"}
    decks = []
    for d in db["deck"].find(flt).sort("_id", -1):
        did = str(d["_id"])
        count = db["flashcard"].count_documents({"deckId": did})
        decks.append({
            "id": did,
            "name": d.get("name"),
            "description": d.get("description"),
            "totalCards": count,
            "lastStudied": d.get("lastStudied"),
        })
    return decks


@app.post("/api/decks")
async def create_deck(payload: DeckCreate, current_user: dict = Depends(get_current_user)):
    doc = {
        "userId": str(current_user["_id"]),
        "name": payload.name.strip(),
        "description": (payload.description or "").strip(),
        "totalCards": 0,
        "studyTime": 0,
        "lastStudied": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    result = db["deck"].insert_one(doc)
    return {"id": str(result.inserted_id), **{k: doc[k] for k in ["name", "description", "totalCards", "lastStudied"]}}


@app.get("/api/decks/{deck_id}")
async def get_deck(deck_id: str, current_user: dict = Depends(get_current_user)):
    d = db["deck"].find_one({"_id": ObjectId(deck_id), "userId": str(current_user["_id"])})
    if not d:
        raise HTTPException(status_code=404, detail="Deck not found")
    cards = list(db["flashcard"].find({"deckId": deck_id}).sort("_id", 1))
    return {
        "id": deck_id,
        "name": d.get("name"),
        "description": d.get("description"),
        "cards": [
            {"id": str(c["_id"]), "question": c["question"], "answer": c["answer"], "difficulty": c.get("difficulty", "medium")}
            for c in cards
        ]
    }


@app.put("/api/decks/{deck_id}")
async def update_deck(deck_id: str, payload: DeckUpdate, current_user: dict = Depends(get_current_user)):
    updates: Dict[str, Any] = {"updated_at": datetime.utcnow()}
    if payload.name is not None:
        updates["name"] = payload.name.strip()
    if payload.description is not None:
        updates["description"] = payload.description.strip()
    res = db["deck"].update_one({"_id": ObjectId(deck_id), "userId": str(current_user["_id"])}, {"$set": updates})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Deck not found")
    return {"updated": True}


@app.delete("/api/decks/{deck_id}")
async def delete_deck(deck_id: str, current_user: dict = Depends(get_current_user)):
    db["flashcard"].delete_many({"deckId": deck_id, "userId": str(current_user["_id"])})
    res = db["deck"].delete_one({"_id": ObjectId(deck_id), "userId": str(current_user["_id"])})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Deck not found")
    return {"deleted": True}


@app.post("/api/decks/{deck_id}/cards")
async def add_card(deck_id: str, payload: CardCreate, current_user: dict = Depends(get_current_user)):
    d = db["deck"].find_one({"_id": ObjectId(deck_id), "userId": str(current_user["_id"])})
    if not d:
        raise HTTPException(status_code=404, detail="Deck not found")
    card = {
        "deckId": deck_id,
        "userId": str(current_user["_id"]),
        "question": payload.question.strip(),
        "answer": payload.answer.strip(),
        "difficulty": payload.difficulty or "medium",
        "reviewCount": 0,
        "lastReviewed": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    ins = db["flashcard"].insert_one(card)
    db["deck"].update_one({"_id": ObjectId(deck_id)}, {"$inc": {"totalCards": 1}})
    return {"id": str(ins.inserted_id)}


@app.put("/api/cards/{card_id}")
async def update_card(card_id: str, payload: CardUpdate, current_user: dict = Depends(get_current_user)):
    c = db["flashcard"].find_one({"_id": ObjectId(card_id), "userId": str(current_user["_id"])})
    if not c:
        raise HTTPException(status_code=404, detail="Card not found")
    updates: Dict[str, Any] = {"updated_at": datetime.utcnow()}
    for k in ["question", "answer", "difficulty"]:
        v = getattr(payload, k)
        if v is not None:
            updates[k] = v.strip() if isinstance(v, str) else v
    db["flashcard"].update_one({"_id": ObjectId(card_id)}, {"$set": updates})
    return {"updated": True}


@app.delete("/api/cards/{card_id}")
async def delete_card(card_id: str, current_user: dict = Depends(get_current_user)):
    c = db["flashcard"].find_one({"_id": ObjectId(card_id), "userId": str(current_user["_id"])})
    if not c:
        raise HTTPException(status_code=404, detail="Card not found")
    db["flashcard"].delete_one({"_id": ObjectId(card_id)})
    db["deck"].update_one({"_id": ObjectId(c["deckId"])}, {"$inc": {"totalCards": -1}})
    return {"deleted": True}


# ----------------------- AI Generation -----------------------
PROMPT_TEMPLATE = (
    "You are an expert flashcard creator for spaced repetition learning. "
    "Generate high-quality study flashcards from the topic: '{topic}'. "
    "Cover definitions, concepts, scenarios, and practical tips. Include a mix of question types: short-answer, true/false, fill-in-the-blank, and multi-select MCQ. "
    "Return STRICT JSON with the following schema: "
    "[{\"question\": str, \"answer\": str, \"difficulty\": one of ['easy','medium','hard']}]. "
    "Do not include any text outside JSON. Aim for {num_cards} diverse cards."
)


@app.post("/api/generate/topic")
async def generate_from_topic(payload: GenerateTopicRequest, current_user: dict = Depends(get_current_user)):
    # Load user's Gemini key from settings
    settings = db["settings"].find_one({"userId": str(current_user["_id"])}) or {}
    api_key = settings.get("geminiApiKey")
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API key not set in Settings")
    if genai is None:
        raise HTTPException(status_code=500, detail="google-generativeai not available")

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)

    prompt = PROMPT_TEMPLATE.format(topic=payload.topic.strip(), num_cards=int(payload.num_cards))
    try:
        resp = model.generate_content(prompt)
        text = resp.text if hasattr(resp, "text") else str(resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)[:200]}")

    # Parse JSON
    import json
    cards: List[Dict[str, Any]] = []
    try:
        # Try to find JSON in the response
        start = text.find("[")
        end = text.rfind("]") + 1
        json_str = text[start:end] if start != -1 and end != -1 else text
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            for c in parsed:
                q = str(c.get("question", "")).strip()
                a = str(c.get("answer", "")).strip()
                d = (c.get("difficulty") or "medium").lower()
                if q and a:
                    cards.append({"question": q, "answer": a, "difficulty": d if d in ["easy","medium","hard"] else "medium"})
    except Exception:
        # fallback: single card
        cards = [{"question": payload.topic.strip(), "answer": text.strip(), "difficulty": "medium"}]

    if not cards:
        raise HTTPException(status_code=500, detail="No cards generated")

    # Create deck and insert cards
    deck_doc = {
        "userId": str(current_user["_id"]),
        "name": payload.deckName or f"{payload.topic.strip().title()} Deck",
        "description": f"Generated from topic: {payload.topic.strip()}",
        "totalCards": len(cards),
        "studyTime": 0,
        "lastStudied": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    deck_res = db["deck"].insert_one(deck_doc)
    deck_id = str(deck_res.inserted_id)

    for c in cards:
        db["flashcard"].insert_one({
            "deckId": deck_id,
            "userId": str(current_user["_id"]),
            "question": c["question"],
            "answer": c["answer"],
            "difficulty": c.get("difficulty", "medium"),
            "reviewCount": 0,
            "lastReviewed": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        })

    return {"deckId": deck_id, "created": len(cards)}


# ----------------------- Study Sessions -----------------------
class StudyFinishPayload(BaseModel):
    durationSec: int
    results: List[Dict[str, Any]]  # [{cardId, correct: bool}]


@app.post("/api/study/{deck_id}/start")
async def study_start(deck_id: str, current_user: dict = Depends(get_current_user)):
    d = db["deck"].find_one({"_id": ObjectId(deck_id), "userId": str(current_user["_id"])})
    if not d:
        raise HTTPException(status_code=404, detail="Deck not found")
    cards = list(db["flashcard"].find({"deckId": deck_id}).sort("_id", 1))
    return {
        "deck": {"id": deck_id, "name": d.get("name"), "description": d.get("description")},
        "cards": [
            {"id": str(c["_id"]), "question": c["question"], "answer": c["answer"], "difficulty": c.get("difficulty", "medium")}
            for c in cards
        ]
    }


@app.post("/api/study/{deck_id}/finish")
async def study_finish(deck_id: str, payload: StudyFinishPayload, current_user: dict = Depends(get_current_user)):
    uid = str(current_user["_id"]) 
    d = db["deck"].find_one({"_id": ObjectId(deck_id), "userId": uid})
    if not d:
        raise HTTPException(status_code=404, detail="Deck not found")
    # Update deck study time and last studied
    db["deck"].update_one({"_id": ObjectId(deck_id)}, {"$inc": {"studyTime": int(payload.durationSec)}, "$set": {"lastStudied": datetime.utcnow()}})
    # Update card stats
    for r in payload.results:
        try:
            cid = r.get("cardId")
            correct = bool(r.get("correct"))
            db["flashcard"].update_one({"_id": ObjectId(cid), "userId": uid}, {
                "$inc": {"reviewCount": 1},
                "$set": {"lastReviewed": datetime.utcnow()}
            })
        except Exception:
            continue
    # Compute accuracy
    total = len(payload.results)
    correct = sum(1 for r in payload.results if r.get("correct"))
    acc = (correct / total * 100) if total else 0.0
    return {"accuracy": acc, "total": total, "correct": correct}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
