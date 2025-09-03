Perfect — let’s start with data generation (synthetic SRs and known issue signatures) + database setup (SQLite).
This will give you test data to run your notebook pipeline end-to-end.


---

📌 Jupyter Notebook Cells

Cell 1 – Imports & DB setup

import os
import uuid
import json
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker

# --- Database Setup ---
DB_URL = "sqlite:///./rma_vfa.db"
engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# --- Table Definitions ---
class SignatureModel(Base):
    __tablename__ = "signatures"
    signature_id = Column(String, primary_key=True, index=True)
    name = Column(String)
    device_types = Column(Text)  # JSON string
    error_codes = Column(Text)   # JSON string
    signature_hash = Column(String, index=True)
    embedding = Column(Text, nullable=True)  # store as JSON for simplicity
    afr = Column(Float, nullable=True)
    examples = Column(Text)  # JSON
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)

class ClusterModel(Base):
    __tablename__ = "clusters"
    cluster_id = Column(String, primary_key=True, index=True)
    name = Column(String)
    centroid = Column(Text)  # JSON embedding
    members_count = Column(Integer, default=1)
    status = Column(String, default="pre-issue")
    meta = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)

class IssueModel(Base):
    __tablename__ = "issues"
    issue_id = Column(String, primary_key=True, index=True)
    cluster_id = Column(String, index=True)
    name = Column(String)
    status = Column(String)
    first_seen = Column(DateTime)
    last_seen = Column(DateTime)
    summary = Column(Text)
    customer_symptoms = Column(Text)
    impacted_pids = Column(Text)  # JSON
    sn_failed = Column(Text)      # JSON
    predicted_afr = Column(Float)
    owner = Column(String)
    created_by = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class CaseHistoryModel(Base):
    __tablename__ = "cases"
    case_id = Column(String, primary_key=True, index=True)
    issue_id = Column(String, index=True, nullable=True)
    decision_type = Column(String)
    confidence = Column(Float)
    case_signature = Column(Text)   # JSON
    decision_data = Column(Text)    # JSON
    langsmith_run_id = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create all tables
Base.metadata.create_all(bind=engine)
print("✅ Database initialized with tables: signatures, clusters, issues, cases")


---

Cell 2 – Synthetic Data Generators

import random

# --- Synthetic Service Request Generator ---
def generate_synthetic_sr(i: int):
    device_types = ["ISR4451", "ASR1001", "NCS540"]
    error_code_options = [
        ["ERR_PWR_12", "FAN_OVT"],
        ["ERR_MEM", "CPU_HANG"],
        ["PORT_DOWN", "LINK_FLAP"],
    ]
    symptoms_options = [
        "Unit reboots intermittently after boot",
        "High fan speed with thermal shutdown",
        "Interface flaps randomly under load",
    ]

    sr = {
        "sr_id": f"SR-2025-{1000+i}",
        "device_type": random.choice(device_types),
        "firmware": "16.9.1",
        "error_codes": random.choice(error_code_options),
        "obfl": f"system logs for case {i} ...",
        "symptoms": random.choice(symptoms_options),
        "geo": random.choice(["APAC", "EMEA", "AMER"]),
        "attachments": []
    }
    return sr

# --- Synthetic Known Issue Signature Generator ---
def generate_known_issue_signature(name: str, device: str, error_codes: list, examples: list):
    return {
        "signature_id": str(uuid.uuid4()),
        "name": name,
        "device_types": json.dumps([device]),
        "error_codes": json.dumps(error_codes),
        "signature_hash": f"{device}|{'|'.join(error_codes)}",
        "embedding": json.dumps([random.random() for _ in range(10)]),  # stub embeddings
        "afr": round(random.uniform(0.01, 0.10), 3),
        "examples": json.dumps(examples),
        "status": "active",
        "created_at": datetime.utcnow()
    }


---

Cell 3 – Insert Synthetic Data into DB

# Open DB session
db = SessionLocal()

# Insert 3 Known Issue Signatures
known_sigs = [
    generate_known_issue_signature("PSU Thermal Fault", "ISR4451", ["ERR_PWR_12","FAN_OVT"], ["SR-2025-1001"]),
    generate_known_issue_signature("Memory Leak Crash", "ASR1001", ["ERR_MEM","CPU_HANG"], ["SR-2025-1002"]),
    generate_known_issue_signature("Port Flap Issue", "NCS540", ["PORT_DOWN","LINK_FLAP"], ["SR-2025-1003"]),
]

for sig in known_sigs:
    db.merge(SignatureModel(**sig))

db.commit()
print("✅ Inserted synthetic Known Issue Signatures")

# Generate 5 random SRs
synthetic_srs = [generate_synthetic_sr(i) for i in range(5)]
print("✅ Generated sample SRs")
print(json.dumps(synthetic_srs, indent=2))


---

Cell 4 – Verify DB Contents

# Check what’s inside DB
signatures = db.query(SignatureModel).all()
print("Known Issue Signatures in DB:")
for s in signatures:
    print(f"- {s.name} | Device: {s.device_types} | Errors: {s.error_codes}")

db.close()


---

✅ After running these 4 cells:

You’ll have a SQLite DB (rma_vfa.db) with the required tables.

It will contain 3 known issue signatures.

You’ll have a generator to create synthetic SR data on the fly for testing the pipeline.



---

👉 Next, we’ll wire this DB + synthetic data into Cluster 1 (Intake & Validation) so the pipeline can start consuming SRs.

Do you want me to now give you the Cluster 1 agents code (Intake + Validation) that consumes these synthetic SRs and prepares CaseSignature objects?

