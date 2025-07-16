#!/usr/bin/env python3
"""
Build knowledge graph from processed documents.
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "backend"))

from knowledge_graph.construction.graph_builder import GraphBuilder
from knowledge_graph.construction.entity_extractor import EntityExtractor
from knowledge_graph.construction.relation_extractor import RelationExtractor
from services.neo4j_client import Neo4jClient
from utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Main knowledge graph building function."""
    parser = argparse.ArgumentParser(description="Build knowledge graph")
    parser.add_argument("--input", required=True, help="Input processed JSON file")
    parser.add_argument("--clear-db", action="store_true", help="Clear existing database")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Load input data
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    
    logger.info(f"Loaded {len(processed_data)} processed items")
    
    # Initialize Neo4j client
    neo4j_client = Neo4jClient()
    
    try:
        # Clear database if requested
        if args.clear_db:
            logger.info("Clearing existing database...")
            await neo4j_client.clear_database()
        
        # Create indexes and constraints
        await neo4j_client.create_index("Entity", "name")
        await neo4j_client.create_index("Entity", "type")
        await neo4j_client.create_constraint("Entity", "name", "UNIQUE")
        
        # Initialize graph builder
        graph_builder = GraphBuilder(neo4j_client)
        
        # Build knowledge graph
        await graph_builder.build_from_documents(processed_data, args.batch_size)
        
        # Get statistics
        stats = await neo4j_client.get_node_statistics()
        logger.info(f"Knowledge graph built successfully: {stats}")
        
    finally:
        await neo4j_client.close()


if __name__ == "__main__":
    asyncio.run(main())
