import os
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Protocol
from datetime import datetime, timezone
from llama_cloud import LlamaCloud

from app.services.file_service import FileService
from app.services.event_service import IEventService
from app.db import TenantCollection
from app.models.file_parse import FileParseRecord

class IParseService(Protocol):
    async def parse_file(self, file_id: str, org_id: str) -> Dict[str, Any]:
        """
        Parses a file and returns the result.
        """
        ...

class LlamaParseService(IParseService):
    def __init__(
        self, 
        parse_collection: TenantCollection, 
        file_service: FileService, 
        event_service: IEventService
    ):
        self.parse_collection = parse_collection
        self.file_service = file_service
        self.event_service = event_service
        
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY must be set")
            
        self.client = LlamaCloud(api_key=api_key)
        self.tier = "cost_effective"
        self.version = "latest"

    async def parse_file(self, file_id: str, org_id: str) -> Dict[str, Any]:
        # Get file metadata
        file_res = await self.file_service.get_file(file_id)
        if not file_res:
            raise ValueError(f"File {file_id} not found")

        # Download file from Supabase
        try:
            file_data = await self.file_service.download_file(file_id)
        except Exception as e:
            raise RuntimeError(f"Failed to download file from file service: {str(e)}")

        # Call LlamaParse v2
        try:
            extension = file_res.extension
            with tempfile.NamedTemporaryFile(delete=True, suffix=f".{extension}") as tmp:
                tmp.write(file_data)
                tmp.flush()
                
                result = self.client.parsing.parse(
                    upload_file=tmp.name,
                    tier=self.tier,
                    version=self.version,
                    processing_options={
                        "ignore": {
                            "ignore_diagonal_text": True
                        }
                    },
                    output_options={
                        "markdown": {
                            "tables": {
                                "merge_continued_tables": True
                            }
                        }
                    },
                    expand=["markdown", "text", "items"]
                )
        except Exception as e:
            raise RuntimeError(f"LlamaParse call failed: {str(e)}")

        # Save result in MongoDB as BSON (Python dict)
        if hasattr(result, "model_dump"):
            parse_result = result.model_dump()
        elif hasattr(result, "dict"):
            parse_result = result.dict()
        else:
            # Fallback for complex objects that might not have dict/model_dump but are still dictable
            parse_result = vars(result)

        parse_record = FileParseRecord(
            file_id=file_id,
            org_id=org_id,
            result=parse_result,
            metadata={
                "parser": "llama_parse_v2",
                "tier": self.tier,
                "version": self.version
            }
        )
        
        self.parse_collection.insert_one(parse_record.model_dump())

        # Push event
        await self.event_service.push("file:parsed", {
            "file_id": file_id,
            "org_id": org_id,
            "parser": "llama_parse_v2"
        })

        return parse_record.model_dump()
