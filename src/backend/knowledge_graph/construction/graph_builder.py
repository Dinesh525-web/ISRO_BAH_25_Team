"""
Knowledge graph builder for MOSDAC domain.
"""
import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from ..ontology.mosdac_ontology import MOSDACOntology
from .entity_extractor import EntityExtractor
from .relation_extractor import RelationExtractor
from app.services.neo4j_client import Neo4jClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GraphBuilder:
    """Build knowledge graph from processed documents."""
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j_client = neo4j_client
        self.ontology = MOSDACOntology()
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
        
        # Track created entities to avoid duplicates
        self.created_entities: Set[str] = set()
        self.entity_id_map: Dict[str, str] = {}
    
    async def build_from_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from processed documents.
        
        Args:
            documents: List of processed documents
            batch_size: Batch size for processing
            
        Returns:
            Build statistics
        """
        logger.info(f"Building knowledge graph from {len(documents)} documents")
        
        total_entities = 0
        total_relations = 0
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            # Extract entities and relations from batch
            batch_entities = []
            batch_relations = []
            
            for doc in batch:
                try:
                    # Extract entities from document
                    entities = await self.entity_extractor.extract_entities(doc)
                    batch_entities.extend(entities)
                    
                    # Extract relations from document
                    relations = await self.relation_extractor.extract_relations(doc, entities)
                    batch_relations.extend(relations)
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc.get('id', 'unknown')}: {e}")
                    continue
            
            # Create entities in Neo4j
            created_entities = await self._create_entities_batch(batch_entities)
            total_entities += len(created_entities)
            
            # Create relations in Neo4j
            created_relations = await self._create_relations_batch(batch_relations)
            total_relations += len(created_relations)
            
            logger.info(f"Batch completed: {len(created_entities)} entities, {len(created_relations)} relations")
        
        # Build final statistics
        stats = {
            'total_documents_processed': len(documents),
            'total_entities_created': total_entities,
            'total_relations_created': total_relations,
            'processing_time': datetime.utcnow().isoformat(),
            'graph_statistics': await self.neo4j_client.get_node_statistics()
        }
        
        logger.info(f"Knowledge graph building completed: {stats}")
        return stats
    
    async def _create_entities_batch(self, entities: List[Dict[str, Any]]) -> List[str]:
        """Create entities in Neo4j in batch."""
        created_ids = []
        
        for entity in entities:
            try:
                # Check if entity already exists
                entity_key = f"{entity['name']}:{entity['type']}"
                
                if entity_key in self.created_entities:
                    continue
                
                # Create entity in Neo4j
                entity_id = await self.neo4j_client.create_node(
                    label="Entity",
                    properties={
                        'name': entity['name'],
                        'type': entity['type'],
                        'description': entity.get('description', ''),
                        'aliases': entity.get('aliases', []),
                        'confidence': entity.get('confidence', 1.0),
                        'source_documents': entity.get('source_documents', []),
                        'properties': entity.get('properties', {}),
                        'created_at': datetime.utcnow().isoformat(),
                        'updated_at': datetime.utcnow().isoformat()
                    },
                    unique_key='name'
                )
                
                # Track created entity
                self.created_entities.add(entity_key)
                self.entity_id_map[entity_key] = entity_id
                created_ids.append(entity_id)
                
            except Exception as e:
                logger.error(f"Error creating entity {entity.get('name', 'unknown')}: {e}")
                continue
        
        return created_ids
    
    async def _create_relations_batch(self, relations: List[Dict[str, Any]]) -> List[str]:
        """Create relations in Neo4j in batch."""
        created_ids = []
        
        for relation in relations:
            try:
                # Get entity IDs
                source_key = f"{relation['source_entity']}:{relation['source_type']}"
                target_key = f"{relation['target_entity']}:{relation['target_type']}"
                
                source_id = self.entity_id_map.get(source_key)
                target_id = self.entity_id_map.get(target_key)
                
                if not source_id or not target_id:
                    logger.warning(f"Missing entity IDs for relation: {relation}")
                    continue
                
                # Create relation in Neo4j
                relation_id = await self.neo4j_client.create_relationship(
                    source_node_id=source_id,
                    target_node_id=target_id,
                    relationship_type=relation['relation_type'],
                    properties={
                        'confidence': relation.get('confidence', 1.0),
                        'source_documents': relation.get('source_documents', []),
                        'context': relation.get('context', ''),
                        'properties': relation.get('properties', {}),
                        'created_at': datetime.utcnow().isoformat()
                    }
                )
                
                created_ids.append(relation_id)
                
            except Exception as e:
                logger.error(f"Error creating relation {relation.get('relation_type', 'unknown')}: {e}")
                continue
        
        return created_ids
    
    async def enrich_with_external_data(self, external_sources: List[str]):
        """Enrich knowledge graph with external data sources."""
        logger.info("Enriching knowledge graph with external data")
        
        # This would integrate with external APIs and databases
        # For now, we'll implement a stub
        
        enrichment_stats = {
            'external_sources_processed': len(external_sources),
            'entities_enriched': 0,
            'relations_added': 0
        }
        
        return enrichment_stats
    
    async def validate_graph_consistency(self) -> Dict[str, Any]:
        """Validate knowledge graph consistency."""
        logger.info("Validating knowledge graph consistency")
        
        validation_results = {
            'total_entities': 0,
            'total_relations': 0,
            'orphaned_entities': 0,
            'duplicate_entities': 0,
            'invalid_relations': 0,
            'consistency_score': 0.0
        }
        
        try:
            # Get graph statistics
            stats = await self.neo4j_client.get_node_statistics()
            validation_results['total_entities'] = stats.get('total_nodes', 0)
            validation_results['total_relations'] = stats.get('total_relationships', 0)
            
            # Check for orphaned entities
            orphaned_query = """
            MATCH (n:Entity)
            WHERE NOT (n)--()
            RETURN count(n) as orphaned_count
            """
            orphaned_result = await self.neo4j_client.execute_query(orphaned_query, fetch_all=False)
            validation_results['orphaned_entities'] = orphaned_result.get('orphaned_count', 0)
            
            # Check for duplicate entities
            duplicate_query = """
            MATCH (n:Entity)
            WITH n.name as name, count(n) as count
            WHERE count > 1
            RETURN sum(count) as duplicate_count
            """
            duplicate_result = await self.neo4j_client.execute_query(duplicate_query, fetch_all=False)
            validation_results['duplicate_entities'] = duplicate_result.get('duplicate_count', 0)
            
            # Calculate consistency score
            total_issues = (
                validation_results['orphaned_entities'] +
                validation_results['duplicate_entities'] +
                validation_results['invalid_relations']
            )
            
            if validation_results['total_entities'] > 0:
                validation_results['consistency_score'] = max(
                    0.0,
                    1.0 - (total_issues / validation_results['total_entities'])
                )
            
        except Exception as e:
            logger.error(f"Error validating graph consistency: {e}")
        
        return validation_results
