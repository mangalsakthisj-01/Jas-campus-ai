from pydantic import BaseModel, EmailStr, Field, field_validator

class RegisterRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=160)
    password: str = Field(min_length=10, max_length=128)

    @field_validator("full_name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Name is required")
        return value

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)

class ManagerCreateRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=160)
    password: str = Field(min_length=12, max_length=128)

class UserDecisionRequest(BaseModel):
    status: str

    @field_validator("status")
    @classmethod
    def check_status(cls, value: str) -> str:
        if value not in {"approved", "rejected", "disabled"}:
            raise ValueError("status must be approved, rejected, or disabled")
        return value

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    conversation_id: int | None = None
    attachment_ids: list[int] = Field(default_factory=list, max_length=8)
    mode: str = Field(default="general", max_length=30)

    @field_validator("mode")
    @classmethod
    def check_mode(cls, value: str) -> str:
        allowed = {"general", "study", "code", "assignment", "decision"}
        if value not in allowed:
            raise ValueError("Unsupported chat mode")
        return value

class DocumentRequest(BaseModel):
    title: str = Field(default="JAS Document", min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=100000)
    format: str = "pdf"

    @field_validator("format")
    @classmethod
    def check_format(cls, value: str) -> str:
        value = value.lower()
        if value not in {"pdf", "docx"}:
            raise ValueError("format must be pdf or docx")
        return value

class ImageRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=2000)
