import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.error_handler import setup_error_handlers
from app.models.chat import ChatCreateRequest, ChatContinueRequest
from app.models.agent import AgentCreateRequest, AgentUpdateRequest, AddToolsRequest, RemoveToolsRequest
from app.models.auth import LoginRequest, RegisterRequest
from app.models.user import UserCreate, UserUpdate, PasswordUpdate
from app.models.file import FileUploadRequest, FileUpdateRequest, BatchFileCompleteRequest
from app.models.organization import OrgCreate, OrgUpdate, OrgMemberUpdate, OrgIntegrationUpdate
from app.models.rag import RagCreateRequest, RagUpdateRequest
from app.models.tool import ToolRegisterRequest
from app.models.llm import LlmCreateRequest

class TestInputValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_app = FastAPI()

        @cls.test_app.post("/test/chat-create")
        def chat_create(req: ChatCreateRequest):
            return {"status": "ok"}

        @cls.test_app.post("/test/chat-continue")
        def chat_continue(req: ChatContinueRequest):
            return {"status": "ok"}

        @cls.test_app.post("/test/agent-create")
        def agent_create(req: AgentCreateRequest):
            return {"status": "ok"}

        @cls.test_app.post("/test/add-tools")
        def add_tools(req: AddToolsRequest):
            return {"status": "ok"}

        @cls.test_app.post("/test/login")
        def login(req: LoginRequest):
            return {"status": "ok"}

        @cls.test_app.post("/test/register")
        def register(req: RegisterRequest):
            return {"status": "ok"}

        @cls.test_app.post("/test/user-create")
        def user_create(req: UserCreate):
            return {"status": "ok"}

        @cls.test_app.post("/test/user-update")
        def user_update(req: UserUpdate):
            return {"status": "ok"}

        @cls.test_app.post("/test/password-update")
        def password_update(req: PasswordUpdate):
            return {"status": "ok"}

        @cls.test_app.post("/test/file-upload")
        def file_upload(req: FileUploadRequest):
            return {"status": "ok"}

        @cls.test_app.post("/test/batch-file-complete")
        def batch_file_complete(req: BatchFileCompleteRequest):
            return {"status": "ok"}

        @cls.test_app.post("/test/org-create")
        def org_create(req: OrgCreate):
            return {"status": "ok"}

        @cls.test_app.post("/test/org-member-update")
        def org_member_update(req: OrgMemberUpdate):
            return {"status": "ok"}

        @cls.test_app.post("/test/rag-create")
        def rag_create(req: RagCreateRequest):
            return {"status": "ok"}

        @cls.test_app.post("/test/tool-register")
        def tool_register(req: ToolRegisterRequest):
            return {"status": "ok"}

        @cls.test_app.post("/test/llm-create")
        def llm_create(req: LlmCreateRequest):
            return {"status": "ok"}

        setup_error_handlers(cls.test_app)
        cls.client = TestClient(cls.test_app, raise_server_exceptions=False)

    def test_chat_message_length_boundaries(self):
        # Valid
        res = self.client.post("/test/chat-create", json={"message": "Hello"})
        self.assertEqual(res.status_code, 200)

        # Invalid: empty
        res = self.client.post("/test/chat-create", json={"message": ""})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json()["error"]["code"], "SYSTEM_VALIDATION_ERROR")

        # Invalid: too long (10001 characters)
        long_msg = "a" * 10001
        res = self.client.post("/test/chat-create", json={"message": long_msg})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json()["error"]["code"], "SYSTEM_VALIDATION_ERROR")

        # Continue request validation
        res = self.client.post("/test/chat-continue", json={"message": ""})
        self.assertEqual(res.status_code, 422)

    def test_file_upload_size_limits(self):
        # Valid: 5MB
        res = self.client.post("/test/file-upload", json={"filename": "doc.pdf", "size": 5 * 1024 * 1024})
        self.assertEqual(res.status_code, 200)

        # Invalid: 0 bytes
        res = self.client.post("/test/file-upload", json={"filename": "doc.pdf", "size": 0})
        self.assertEqual(res.status_code, 422)

        # Invalid: 51MB (> 50MB limit)
        res = self.client.post("/test/file-upload", json={"filename": "doc.pdf", "size": 51 * 1024 * 1024})
        self.assertEqual(res.status_code, 422)

        # Invalid filename empty
        res = self.client.post("/test/file-upload", json={"filename": "", "size": 100})
        self.assertEqual(res.status_code, 422)

    def test_auth_validation(self):
        # Invalid email
        res = self.client.post("/test/login", json={"email": "not-an-email", "password": "securepassword"})
        self.assertEqual(res.status_code, 422)

        # Password too short
        res = self.client.post("/test/login", json={"email": "test@example.com", "password": "123"})
        self.assertEqual(res.status_code, 422)

        # Register request invalid first_name/last_name
        res = self.client.post("/test/register", json={
            "email": "test@example.com",
            "password": "securepassword",
            "first_name": "",
            "last_name": "Doe"
        })
        self.assertEqual(res.status_code, 422)

    def test_agent_creation_limits(self):
        # Invalid name empty
        res = self.client.post("/test/agent-create", json={
            "name": "",
            "role": "Assistant",
            "goal": "Help people",
            "backstory": "Created to help",
            "llm_id": "llm-1"
        })
        self.assertEqual(res.status_code, 422)

        # Invalid goal too long
        res = self.client.post("/test/agent-create", json={
            "name": "Helper",
            "role": "Assistant",
            "goal": "a" * 1001,
            "backstory": "Created to help",
            "llm_id": "llm-1"
        })
        self.assertEqual(res.status_code, 422)

    def test_rag_limits(self):
        # Valid
        res = self.client.post("/test/rag-create", json={"title": "Doc", "content": "Sample content"})
        self.assertEqual(res.status_code, 200)

        # Invalid title empty
        res = self.client.post("/test/rag-create", json={"title": "", "content": "Sample content"})
        self.assertEqual(res.status_code, 422)

        # Invalid content too long (> 50k characters)
        res = self.client.post("/test/rag-create", json={"title": "Doc", "content": "a" * 50001})
        self.assertEqual(res.status_code, 422)
