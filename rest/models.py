from sqlalchemy import create_engine, Column, String, Integer, DateTime, Float, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os
import uuid  # Add this import for UUID generation

Base = declarative_base()

class Video(Base):
    __tablename__ = 'videos'
    
    id = Column(Integer, primary_key=True)
    url = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=True)
    category = Column(String, nullable=False)
    download_date = Column(DateTime, default=datetime.datetime.utcnow)
    file_path = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending, success, failed
    duration = Column(Integer, nullable=True)  # in seconds
    resolution = Column(String, nullable=True)  # e.g. "1080p"
    video_bitrate = Column(Float, nullable=True)  # in kbps
    audio_bitrate = Column(Float, nullable=True)  # in kbps
    error_message = Column(String, nullable=True)

class CandidateVideo(Base):
    __tablename__ = 'candidate_videos'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))  # Changed from Integer to String with UUID default
    video_id = Column(String, nullable=False)  # YouTube video ID
    url = Column(String, nullable=False)
    title = Column(String, nullable=True)
    category = Column(String, nullable=False)
    description = Column(String, nullable=True)
    duration = Column(Integer, nullable=True)  # in seconds
    height = Column(Integer, nullable=True)    # resolution height
    thumbnail = Column(String, nullable=True)  # thumbnail URL
    vbr = Column(Float, nullable=True)  # video bitrate
    abr = Column(Float, nullable=True)  # audio bitrate
    view_count = Column(Integer, nullable=True)
    creation_date = Column(DateTime, default=datetime.datetime.utcnow)
    last_download_date = Column(DateTime, nullable=True)  # When the candidate was last downloaded
    validation_message = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending, approved, rejected

class Image(Base):
    __tablename__ = 'images'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    url = Column(String, nullable=False)
    title = Column(String, nullable=True)
    category = Column(String, nullable=False)
    download_date = Column(DateTime, default=datetime.datetime.utcnow)
    file_path = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending, success, failed
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)  # in bytes
    error_message = Column(String, nullable=True)
    thumbnail = Column(String, nullable=True)  # thumbnail URL

# SQLite database setup
db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'video_history.db')
engine = create_engine(f'sqlite:///{db_path}')

# Create tables
Base.metadata.create_all(engine)

# Migration function to add new columns to existing tables
def check_and_update_schema():
    """Check for missing columns and add them if needed"""
    try:
        connection = engine.connect()
        inspector = inspect(engine)
        
        # Check if the candidate_videos table exists
        if 'candidate_videos' in inspector.get_table_names():
            # Get existing columns
            columns = [col['name'] for col in inspector.get_columns('candidate_videos')]
            
            # Check for last_download_date column
            if 'last_download_date' not in columns:
                print("Adding missing column 'last_download_date' to candidate_videos table")
                with connection.begin():
                    connection.execute(text("ALTER TABLE candidate_videos ADD COLUMN last_download_date TIMESTAMP"))
                print("Schema migration completed successfully")
        
        # Check if the images table exists
        if 'images' not in inspector.get_table_names():
            print("Creating images table")
            Base.metadata.create_all(engine, tables=[Image.__table__])
            print("Images table created successfully")
        
        connection.close()
    except Exception as e:
        print(f"Error during schema migration: {e}")

# Run schema migration check
check_and_update_schema()

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_session():
    """Returns a new database session"""
    return SessionLocal()
