import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables from the .env file in the parent directories
# Relative path to .env file from this backend/app/database location
# We check the current working directory first, then look up a few levels.
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cyclone_local.db")

# Connection Arguments
# SQLite requires 'check_same_thread: False' to allow multi-threaded requests in FastAPI
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# Create SQLAlchemy database engine
engine = create_engine(DATABASE_URL, connect_args=connect_args)

# Create SessionLocal class for database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base class for models
Base = declarative_base()

# Dependency to get db session in FastAPI endpoints
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
