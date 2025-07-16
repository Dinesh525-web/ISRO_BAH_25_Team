#!/usr/bin/env python3
"""
Process scraped documents through the data pipeline.
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "backend"))

from data_ingestion.processors.data_processor import DataProcessor
from data_ingestion.processors.text_processor import TextProcessor
from services.database import DatabaseService
from services.redis_client import RedisClient
from models.database import get_db_session
from utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description="Process scraped documents")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Load input data
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        scraped_data = json.load(f)
    
    logger.info(f"Loaded {len(scraped_data)} items from {input_file}")
    
    # Initialize services
    redis_client = RedisClient()
    
    # Get database session
    async with get_db_session() as db:
        db_service = DatabaseService(db)
        
        # Initialize processor
        processor = DataProcessor(db_service, redis_client)
        
        # Process data in batches
        processed_data = await processor.batch_process(scraped_data, args.batch_size)
        
        # Deduplicate
        processed_data = await processor.deduplicate_content(processed_data)
        
        # Filter by quality
        processed_data = await processor.filter_by_quality(processed_data)
        
        # Enrich with metadata
        processed_data = await processor.enrich_with_metadata(processed_data)
        
        logger.info(f"Processed {len(processed_data)} items")
        
        # Save processed data
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"processed_{input_file.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Processed data saved to {output_file}")
    
    await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
