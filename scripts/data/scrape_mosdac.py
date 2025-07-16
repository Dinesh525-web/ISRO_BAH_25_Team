#!/usr/bin/env python3
"""
Standalone script to scrape MOSDAC website.
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "backend"))

from data_ingestion.scrapers.mosdac_scraper import MOSDACscraper
from data_ingestion.scrapers.realtime_scraper import RealtimeScraper
from utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Main scraping function."""
    parser = argparse.ArgumentParser(description="Scrape MOSDAC website")
    parser.add_argument("--type", choices=["static", "realtime", "both"], default="both", 
                       help="Type of scraping to perform")
    parser.add_argument("--max-pages", type=int, default=100, 
                       help="Maximum pages to scrape")
    parser.add_argument("--output", default="data/raw/scraped_data", 
                       help="Output directory")
    parser.add_argument("--sections", nargs="+", 
                       help="Specific sections to scrape")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scraped_data = []
    
    if args.type in ["static", "both"]:
        logger.info("Starting static content scraping...")
        
        async with MOSDACscraper() as scraper:
            if args.sections:
                data = await scraper.scrape_specific_sections(args.sections)
            else:
                data = await scraper.scrape_all(max_pages=args.max_pages)
            
            scraped_data.extend(data)
            
            # Save statistics
            stats = scraper.get_statistics()
            logger.info(f"Static scraping completed: {stats}")
    
    if args.type in ["realtime", "both"]:
        logger.info("Starting realtime content scraping...")
        
        async with RealtimeScraper() as scraper:
            data = await scraper.scrape_all(max_pages=args.max_pages // 2)
            scraped_data.extend(data)
            
            # Save statistics
            stats = scraper.get_statistics()
            logger.info(f"Realtime scraping completed: {stats}")
    
    # Save scraped data
    output_file = output_dir / f"mosdac_scraped_{args.type}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scraped_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Scraped data saved to {output_file}")
    logger.info(f"Total items scraped: {len(scraped_data)}")


if __name__ == "__main__":
    asyncio.run(main())
