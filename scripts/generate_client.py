import os
import sys
from datetime import datetime, timezone
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient
from cryptography.fernet import Fernet

def main():
    if len(sys.argv) < 2:
        print("Error: Missing client ID.")
        print("Usage: python scripts/generate_client.py <client_id>")
        sys.exit(1)
        
    client_id = sys.argv[1].strip()
    if not client_id:
        print("Error: Client ID cannot be empty.")
        sys.exit(1)
        
    env_file = ".env.local"
    # Load env files
    load_dotenv(env_file)
    load_dotenv()
    
    # Check if API_KEY_ENCRYPTION_KEY is set
    encryption_key = os.getenv("API_KEY_ENCRYPTION_KEY")
    if not encryption_key:
        print("API_KEY_ENCRYPTION_KEY not found in environment. Generating a new one...")
        encryption_key = Fernet.generate_key().decode('utf-8')
        # Append to .env.local
        with open(env_file, "a") as f:
            f.write(f"\nAPI_KEY_ENCRYPTION_KEY={encryption_key}\n")
        print(f"Generated new encryption key and saved it to {env_file}")
        os.environ["API_KEY_ENCRYPTION_KEY"] = encryption_key
        
    # Connect to MongoDB
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    print(f"Connecting to MongoDB...")
    
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client["kita_admin"]
    collection = db["clients"]
    
    # Generate API key (which is a valid ObjectId)
    api_key_id = ObjectId()
    api_key_str = str(api_key_id)
    
    # Encrypt the API key
    try:
        fernet = Fernet(encryption_key.encode('utf-8'))
        encrypted_api_key = fernet.encrypt(api_key_str.encode('utf-8')).decode('utf-8')
    except Exception as e:
        print(f"Failed to encrypt API key: {e}")
        sys.exit(1)
        
    # Check if client already exists
    existing = collection.find_one({"client_id": client_id})
    if existing:
        print(f"Client '{client_id}' already exists in DB. Deleting old record to overwrite...")
        collection.delete_one({"client_id": client_id})
        
    # Insert new client record
    collection.insert_one({
        "_id": encrypted_api_key,
        "client_id": client_id,
        "created_at": datetime.now(timezone.utc)
    })
    
    print("\n" + "=" * 50)
    print("CLIENT CREDENTIALS GENERATED SUCCESSFULLY")
    print("=" * 50)
    print(f"x-client-id : {client_id}")
    print(f"x-api-key   : {api_key_str}")
    print("=" * 50)
    print("Store these credentials securely. The API key cannot be retrieved from the DB again.")
    print("=" * 50)

if __name__ == "__main__":
    main()
