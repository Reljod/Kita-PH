from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
from bson import ObjectId
from app.db import Database
from app.models.organization import (
    OrganizationDocument, OrgCreate, OrgUpdate, OrganizationResponse, 
    OrgMember, OrgRole, OrgMemberUpdate, OrgIntegrationUpdate
)

class OrganizationService:
    def __init__(self):
        self.collection = Database.get_organizations_collection()

    def create_org(self, org_in: OrgCreate, creator_id: str) -> OrganizationResponse:
        org_doc = OrganizationDocument(
            org_name=org_in.org_name,
            org_code=org_in.org_code,
            org_members=[OrgMember(user_id=creator_id, role=OrgRole.ADMIN)],
            status="initializing",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        org_dict = org_doc.model_dump()
        result = self.collection.insert_one(org_dict)
        org_dict["id"] = str(result.inserted_id)
        return OrganizationResponse(**org_dict)

    def update_org_status(self, org_id: str, status: str) -> bool:
        try:
            oid = ObjectId(org_id)
            query = {"_id": oid}
        except Exception:
            query = {"_id": org_id}
        result = self.collection.update_one(
            query,
            {"$set": {"status": status, "updated_at": datetime.now(timezone.utc)}}
        )
        return result.modified_count > 0


    def get_org(self, org_id: str) -> Optional[OrganizationResponse]:
        if not org_id:
            return None
            
        # Try finding by ObjectId first
        org = None
        try:
            oid = ObjectId(org_id)
            org = self.collection.find_one({"_id": oid})
        except Exception:
            pass
            
        # If not found by ObjectId, try finding by string ID
        if not org:
            org = self.collection.find_one({"_id": org_id})
            
        if org:
            org["id"] = str(org["_id"])
            return OrganizationResponse(**org)
        return None

    def get_org_by_code(self, org_code: str) -> Optional[OrganizationResponse]:
        org = self.collection.find_one({"org_code": org_code})
        if org:
            org["id"] = str(org["_id"])
            return OrganizationResponse(**org)
        return None

    def get_org_by_integration_id(self, integration_type: str, integration_id: str) -> Optional[OrganizationResponse]:
        org = self.collection.find_one({f"integrations.{integration_type}": integration_id})
        if org:
            org["id"] = str(org["_id"])
            return OrganizationResponse(**org)
        return None

    def get_org_by_id_or_code(self, identifier: str) -> Optional[OrganizationResponse]:
        # Try ID first
        org = self.get_org(identifier)
        if org:
            return org
        # Then try code
        return self.get_org_by_code(identifier)

    def update_org(self, org_id: str, org_in: OrgUpdate) -> Optional[OrganizationResponse]:
        update_data = org_in.model_dump(exclude_unset=True)
        if not update_data:
            return self.get_org(org_id)
            
        update_data["updated_at"] = datetime.now(timezone.utc)
        
        result = self.collection.find_one_and_update(
            {"_id": ObjectId(org_id)},
            {"$set": update_data},
            return_document=True
        )
        if result:
            result["id"] = str(result["_id"])
            return OrganizationResponse(**result)
        return None

    def update_organization_config(self, org_id: str, config: Dict[str, Any]) -> Optional[OrganizationResponse]:
        if not config:
            return self.get_org(org_id)

        set_data = {"updated_at": datetime.now(timezone.utc)}
        for k, v in config.items():
            set_data[f"config.{k}"] = v

        result = self.collection.find_one_and_update(
            {"_id": ObjectId(org_id)},
            {"$set": set_data},
            return_document=True
        )
        if result:
            result["id"] = str(result["_id"])
            return OrganizationResponse(**result)
        return None

    def update_integrations(self, org_id: str, integration_in: OrgIntegrationUpdate) -> Optional[OrganizationResponse]:
        update_data = integration_in.model_dump(exclude_unset=True)
        if not update_data:
            return self.get_org(org_id)
            
        set_data = {"updated_at": datetime.now(timezone.utc)}
        for k, v in update_data.items():
            set_data[f"integrations.{k}"] = v
            
        result = self.collection.find_one_and_update(
            {"_id": ObjectId(org_id)},
            {"$set": set_data},
            return_document=True
        )
        if result:
            result["id"] = str(result["_id"])
            return OrganizationResponse(**result)
        return None

    def add_or_update_member(self, org_id: str, member_in: OrgMemberUpdate) -> Optional[OrganizationResponse]:
        # Check if member already exists
        org = self.collection.find_one({"_id": ObjectId(org_id)})
        if not org:
            return None
        
        members = [OrgMember(**m) for m in org.get("org_members", [])]
        existing_member = next((m for m in members if m.user_id == member_in.user_id), None)
        
        if existing_member:
            # Update role
            self.collection.update_one(
                {"_id": ObjectId(org_id), "org_members.user_id": member_in.user_id},
                {"$set": {"org_members.$.role": member_in.role, "updated_at": datetime.now(timezone.utc)}}
            )
        else:
            # Add member
            self.collection.update_one(
                {"_id": ObjectId(org_id)},
                {"$push": {"org_members": member_in.model_dump()}, "$set": {"updated_at": datetime.now(timezone.utc)}}
            )
            
        return self.get_org(org_id)

    def revoke_member(self, org_id: str, user_id: str) -> Optional[OrganizationResponse]:
        result = self.collection.find_one_and_update(
            {"_id": ObjectId(org_id)},
            {"$pull": {"org_members": {"user_id": user_id}}, "$set": {"updated_at": datetime.now(timezone.utc)}},
            return_document=True
        )
        if result:
            result["id"] = str(result["_id"])
            return OrganizationResponse(**result)
        return None

    def get_user_orgs(self, user_id: str) -> List[OrganizationResponse]:
        orgs = self.collection.find({"org_members.user_id": user_id})
        res = []
        for org in orgs:
            org["id"] = str(org["_id"])
            res.append(OrganizationResponse(**org))
        return res
