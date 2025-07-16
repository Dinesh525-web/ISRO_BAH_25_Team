"""
Cypher query implementations for Neo4j knowledge graph operations.
"""
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import time
import json

from ..ontology.schemas import (
    KGQuery, KGQueryResult, KGEntity, KGRelation, KGPath, 
    EntityTypeEnum, RelationTypeEnum, KGStatistics
)
from ..ontology.mosdac_ontology import MOSDACOntology
from app.services.neo4j_client import Neo4jClient
from app.services.redis_client import RedisClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CypherQueryEngine:
    """Cypher query engine for Neo4j operations."""
    
    def __init__(self, neo4j_client: Neo4jClient, redis_client: Optional[RedisClient] = None):
        self.neo4j_client = neo4j_client
        self.redis_client = redis_client
        self.ontology = MOSDACOntology()
        self.cache_ttl = 300  # 5 minutes
        
        # Query templates
        self.templates = {
            'find_entities': self._build_find_entities_template,
            'find_relations': self._build_find_relations_template,
            'find_paths': self._build_find_paths_template,
            'find_neighbors': self._build_find_neighbors_template,
            'find_similar': self._build_find_similar_template,
            'subgraph_extraction': self._build_subgraph_template,
            'graph_traversal': self._build_traversal_template,
            'aggregation': self._build_aggregation_template,
            'domain_specific': self._build_domain_template,
        }
    
    async def execute_query(self, query_type: str, parameters: Dict[str, Any]) -> KGQueryResult:
        """
        Execute a Cypher query based on type and parameters.
        
        Args:
            query_type: Type of query to execute
            parameters: Query parameters
            
        Returns:
            Query result object
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query_type, parameters)
            cached_result = await self._get_cached_result(cache_key)
            
            if cached_result:
                logger.debug(f"Returning cached result for query type: {query_type}")
                return cached_result
            
            # Build and execute query
            cypher_query, query_params = self._build_query(query_type, parameters)
            
            # Execute query
            raw_results = await self.neo4j_client.execute_query(
                cypher_query, query_params, fetch_all=True
            )
            
            # Process results
            result = await self._process_results(raw_results, query_type, start_time)
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            return KGQueryResult(
                entities=[],
                relations=[],
                paths=[],
                total_count=0,
                query_time=time.time() - start_time,
                query_info={"error": str(e), "query_type": query_type}
            )
    
    def _build_query(self, query_type: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build Cypher query based on type and parameters."""
        if query_type not in self.templates:
            raise ValueError(f"Unsupported query type: {query_type}")
        
        template_builder = self.templates[query_type]
        return template_builder(parameters)
    
    def _build_find_entities_template(self, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build entity finding query."""
        # Base query
        query_parts = ["MATCH (e:Entity)"]
        where_conditions = []
        query_params = {}
        
        # Entity type filter
        if "entity_type" in params:
            where_conditions.append("e.type = $entity_type")
            query_params["entity_type"] = params["entity_type"]
        
        # Name filter
        if "name" in params:
            if params.get("exact_match", False):
                where_conditions.append("e.name = $name")
            else:
                where_conditions.append("e.name CONTAINS $name")
            query_params["name"] = params["name"]
        
        # Property filters
        if "properties" in params:
            for prop_name, prop_value in params["properties"].items():
                condition = f"e.{prop_name} = $prop_{prop_name}"
                where_conditions.append(condition)
                query_params[f"prop_{prop_name}"] = prop_value
        
        # Confidence filter
        if "min_confidence" in params:
            where_conditions.append("e.confidence >= $min_confidence")
            query_params["min_confidence"] = params["min_confidence"]
        
        # Aliases filter
        if "aliases" in params:
            where_conditions.append("$aliases IN e.aliases")
            query_params["aliases"] = params["aliases"]
        
        # Add WHERE clause
        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))
        
        # Add RETURN clause
        query_parts.append("RETURN e, elementId(e) as id")
        
        # Add ordering
        order_by = params.get("order_by", "name")
        order_desc = params.get("order_desc", False)
        query_parts.append(f"ORDER BY e.{order_by} {'DESC' if order_desc else 'ASC'}")
        
        # Add pagination
        offset = params.get("offset", 0)
        limit = params.get("limit", 10)
        query_parts.append(f"SKIP {offset} LIMIT {limit}")
        
        return " ".join(query_parts), query_params
    
    def _build_find_relations_template(self, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build relation finding query."""
        # Base query
        relation_filter = ""
        if "relation_type" in params:
            relation_filter = f":{params['relation_type']}"
        
        query_parts = [f"MATCH (source:Entity)-[r{relation_filter}]->(target:Entity)"]
        where_conditions = []
        query_params = {}
        
        # Source entity filter
        if "source_entity" in params:
            where_conditions.append("source.name = $source_entity")
            query_params["source_entity"] = params["source_entity"]
        
        # Target entity filter
        if "target_entity" in params:
            where_conditions.append("target.name = $target_entity")
            query_params["target_entity"] = params["target_entity"]
        
        # Entity type filters
        if "source_type" in params:
            where_conditions.append("source.type = $source_type")
            query_params["source_type"] = params["source_type"]
        
        if "target_type" in params:
            where_conditions.append("target.type = $target_type")
            query_params["target_type"] = params["target_type"]
        
        # Confidence filter
        if "min_confidence" in params:
            where_conditions.append("r.confidence >= $min_confidence")
            query_params["min_confidence"] = params["min_confidence"]
        
        # Add WHERE clause
        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))
        
        # Add RETURN clause
        query_parts.append(
            "RETURN source, r, target, type(r) as relation_type, "
            "elementId(source) as source_id, elementId(target) as target_id, elementId(r) as relation_id"
        )
        
        # Add ordering
        query_parts.append("ORDER BY source.name, target.name")
        
        # Add pagination
        offset = params.get("offset", 0)
        limit = params.get("limit", 10)
        query_parts.append(f"SKIP {offset} LIMIT {limit}")
        
        return " ".join(query_parts), query_params
    
    def _build_find_paths_template(self, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build path finding query."""
        # Validate required parameters
        if "source_entity" not in params or "target_entity" not in params:
            raise ValueError("Source and target entities are required for path queries")
        
        # Path parameters
        min_length = params.get("min_length", 1)
        max_length = params.get("max_length", 5)
        
        # Build relation type filter
        relation_filter = ""
        if "relation_types" in params:
            relation_types = "|".join(params["relation_types"])
            relation_filter = f":{relation_types}"
        
        # Build query
        query_parts = [
            f"MATCH (source:Entity {{name: $source_entity}})",
            f"MATCH (target:Entity {{name: $target_entity}})",
            f"MATCH path = (source)-[r{relation_filter}*{min_length}..{max_length}]-(target)"
        ]
        
        where_conditions = []
        query_params = {
            "source_entity": params["source_entity"],
            "target_entity": params["target_entity"]
        }
        
        # Exact length filter
        if "exact_length" in params:
            where_conditions.append(f"length(path) = {params['exact_length']}")
        
        # Avoid cycles
        if params.get("avoid_cycles", True):
            where_conditions.append("ALL(n in nodes(path) WHERE size([x in nodes(path) WHERE x = n]) = 1)")
        
        # Add WHERE clause
        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))
        
        # Add RETURN clause
        query_parts.append(
            "RETURN path, length(path) as path_length, "
            "nodes(path) as path_nodes, relationships(path) as path_relations"
        )
        
        # Add ordering
        query_parts.append("ORDER BY path_length")
        
        # Add pagination
        limit = params.get("limit", 10)
        query_parts.append(f"LIMIT {limit}")
        
        return " ".join(query_parts), query_params
    
    def _build_find_neighbors_template(self, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build neighbor finding query."""
        if "entity_name" not in params:
            raise ValueError("Entity name is required for neighbor queries")
        
        # Direction pattern
        direction = params.get("direction", "both")
        if direction == "incoming":
            pattern = "(neighbor:Entity)-[r]->(entity:Entity)"
        elif direction == "outgoing":
            pattern = "(entity:Entity)-[r]->(neighbor:Entity)"
        else:
            pattern = "(entity:Entity)-[r]-(neighbor:Entity)"
        
        query_parts = [
            f"MATCH {pattern}",
            "WHERE entity.name = $entity_name"
        ]
        
        where_conditions = []
        query_params = {"entity_name": params["entity_name"]}
        
        # Neighbor type filter
        if "neighbor_type" in params:
            where_conditions.append("neighbor.type = $neighbor_type")
            query_params["neighbor_type"] = params["neighbor_type"]
        
        # Relation type filter
        if "relation_type" in params:
            where_conditions.append("type(r) = $relation_type")
            query_params["relation_type"] = params["relation_type"]
        
        # Add WHERE conditions
        if where_conditions:
            query_parts.append("AND " + " AND ".join(where_conditions))
        
        # Add RETURN clause
        query_parts.append(
            "RETURN neighbor, r, type(r) as relation_type, "
            "elementId(neighbor) as neighbor_id, elementId(r) as relation_id"
        )
        
        # Add ordering
        query_parts.append("ORDER BY neighbor.name")
        
        # Add pagination
        offset = params.get("offset", 0)
        limit = params.get("limit", 10)
        query_parts.append(f"SKIP {offset} LIMIT {limit}")
        
        return " ".join(query_parts), query_params
    
    def _build_find_similar_template(self, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build similarity finding query."""
        if "entity_name" not in params:
            raise ValueError("Entity name is required for similarity queries")
        
        similarity_method = params.get("similarity_method", "neighbors")
        
        if similarity_method == "neighbors":
            # Structural similarity based on shared neighbors
            query_parts = [
                "MATCH (target:Entity {name: $entity_name})",
                "MATCH (similar:Entity)",
                "WHERE similar.type = target.type AND similar.name <> target.name",
                "OPTIONAL MATCH (target)-[r1]-(neighbor)-[r2]-(similar)",
                "WHERE type(r1) = type(r2)",
                "WITH target, similar, count(DISTINCT neighbor) as shared_neighbors",
                "MATCH (target)-[r1]-(n1), (similar)-[r2]-(n2)",
                "WITH target, similar, shared_neighbors, "
                "count(DISTINCT n1) as target_degree, count(DISTINCT n2) as similar_degree",
                "WITH target, similar, shared_neighbors, target_degree, similar_degree,",
                "CASE WHEN target_degree + similar_degree - shared_neighbors = 0 THEN 0",
                "ELSE toFloat(shared_neighbors) / (target_degree + similar_degree - shared_neighbors) END as similarity"
            ]
        else:
            # Property-based similarity
            query_parts = [
                "MATCH (target:Entity {name: $entity_name})",
                "MATCH (similar:Entity)",
                "WHERE similar.type = target.type AND similar.name <> target.name",
                "WITH target, similar,",
                "CASE WHEN target.type = similar.type THEN 1.0 ELSE 0.0 END as similarity"
            ]
        
        query_params = {"entity_name": params["entity_name"]}
        
        # Minimum similarity threshold
        if "min_similarity" in params:
            query_parts.append("WHERE similarity >= $min_similarity")
            query_params["min_similarity"] = params["min_similarity"]
        
        # Add RETURN clause
        query_parts.append("RETURN similar, similarity, elementId(similar) as id")
        
        # Add ordering
        query_parts.append("ORDER BY similarity DESC")
        
        # Add pagination
        limit = params.get("limit", 10)
        query_parts.append(f"LIMIT {limit}")
        
        return " ".join(query_parts), query_params
    
    def _build_subgraph_template(self, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build subgraph extraction query."""
        query_params = {}
        
        if "center_entity" in params:
            # Ego subgraph
            radius = params.get("radius", 2)
            query_parts = [
                f"MATCH (center:Entity {{name: $center_entity}})",
                f"MATCH path = (center)-[r*1..{radius}]-(node:Entity)",
                "WITH collect(DISTINCT center) + collect(DISTINCT node) as all_nodes",
                "UNWIND all_nodes as n",
                "OPTIONAL MATCH (n)-[r]-(m)",
                "WHERE m IN all_nodes",
                "RETURN collect(DISTINCT n) as nodes, collect(DISTINCT r) as relationships"
            ]
            query_params["center_entity"] = params["center_entity"]
        
        elif "entity_types" in params:
            # Subgraph by entity types
            entity_types = params["entity_types"]
            types_condition = " OR ".join([f"n.type = '{t}'" for t in entity_types])
            query_parts = [
                f"MATCH (n:Entity) WHERE {types_condition}",
                "OPTIONAL MATCH (n)-[r]-(m:Entity)",
                f"WHERE {types_condition.replace('n.type', 'm.type')}",
                "RETURN collect(DISTINCT n) as nodes, collect(DISTINCT r) as relationships"
            ]
        
        else:
            # Full graph
            query_parts = [
                "MATCH (n:Entity)",
                "OPTIONAL MATCH (n)-[r]-(m:Entity)",
                "RETURN collect(DISTINCT n) as nodes, collect(DISTINCT r) as relationships"
            ]
        
        return " ".join(query_parts), query_params
    
    def _build_traversal_template(self, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build graph traversal query."""
        if "start_entity" not in params:
            raise ValueError("Start entity is required for graph traversal")
        
        max_depth = params.get("max_depth", 3)
        
        query_parts = [
            "MATCH (start:Entity {name: $start_entity})",
            f"MATCH path = (start)-[r*1..{max_depth}]-(node:Entity)"
        ]
        
        query_params = {"start_entity": params["start_entity"]}
        
        # Add filters
        where_conditions = []
        
        if "node_types" in params:
            node_types = params["node_types"]
            types_condition = " OR ".join([f"node.type = '{t}'" for t in node_types])
            where_conditions.append(f"({types_condition})")
        
        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))
        
        # Add RETURN clause
        query_parts.append(
            "RETURN DISTINCT node, length(path) as distance, "
            "elementId(node) as id, path"
        )
        
        # Add ordering
        query_parts.append("ORDER BY distance, node.name")
        
        # Add pagination
        limit = params.get("limit", 50)
        query_parts.append(f"LIMIT {limit}")
        
        return " ".join(query_parts), query_params
    
    def _build_aggregation_template(self, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build aggregation query."""
        agg_type = params.get("aggregation_type", "count")
        group_by = params.get("group_by", "type")
        
        query_params = {}
        
        if agg_type == "count":
            if group_by == "type":
                query_parts = [
                    "MATCH (e:Entity)",
                    "RETURN e.type as entity_type, count(e) as count",
                    "ORDER BY count DESC"
                ]
            elif group_by == "relation_type":
                query_parts = [
                    "MATCH ()-[r]-()",
                    "RETURN type(r) as relation_type, count(r) as count",
                    "ORDER BY count DESC"
                ]
        
        elif agg_type == "degree":
            query_parts = [
                "MATCH (e:Entity)",
                "OPTIONAL MATCH (e)-[r]-()",
                "RETURN e.name as entity_name, e.type as entity_type, count(r) as degree",
                "ORDER BY degree DESC"
            ]
        
        elif agg_type == "avg_confidence":
            query_parts = [
                "MATCH (e:Entity)",
                "RETURN e.type as entity_type, avg(e.confidence) as avg_confidence",
                "ORDER BY avg_confidence DESC"
            ]
        
        # Add pagination
        limit = params.get("limit", 20)
        query_parts.append(f"LIMIT {limit}")
        
        return " ".join(query_parts), query_params
    
    def _build_domain_template(self, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build domain-specific queries for MOSDAC."""
        domain_type = params.get("domain_type", "satellite_instruments")
        query_params = {}
        
        if domain_type == "satellite_instruments":
            if "satellite_name" not in params:
                raise ValueError("Satellite name is required")
            
            query_parts = [
                "MATCH (satellite:Entity {type: 'satellite'})-[r:has_instrument]->(instrument:Entity {type: 'instrument'})",
                "WHERE satellite.name = $satellite_name",
                "RETURN satellite, instrument, r,",
                "elementId(satellite) as satellite_id,",
                "elementId(instrument) as instrument_id,",
                "elementId(r) as relation_id",
                "ORDER BY instrument.name"
            ]
            query_params["satellite_name"] = params["satellite_name"]
        
        elif domain_type == "instrument_parameters":
            if "instrument_name" not in params:
                raise ValueError("Instrument name is required")
            
            query_parts = [
                "MATCH (instrument:Entity {type: 'instrument'})-[r:measures]->(parameter:Entity {type: 'parameter'})",
                "WHERE instrument.name = $instrument_name",
                "RETURN instrument, parameter, r,",
                "elementId(instrument) as instrument_id,",
                "elementId(parameter) as parameter_id,",
                "elementId(r) as relation_id",
                "ORDER BY parameter.name"
            ]
            query_params["instrument_name"] = params["instrument_name"]
        
        elif domain_type == "data_chain":
            if "satellite_name" not in params:
                raise ValueError("Satellite name is required")
            
            query_parts = [
                "MATCH path = (satellite:Entity {type: 'satellite'})-[r1:has_instrument]->(instrument:Entity {type: 'instrument'})",
                "-[r2:measures]->(parameter:Entity {type: 'parameter'})",
                "-[r3:used_for]->(application:Entity {type: 'application'})",
                "WHERE satellite.name = $satellite_name",
                "RETURN path, satellite, instrument, parameter, application, r1, r2, r3",
                "ORDER BY instrument.name, parameter.name, application.name"
            ]
            query_params["satellite_name"] = params["satellite_name"]
        
        return " ".join(query_parts), query_params
    
    async def _process_results(
        self,
        raw_results: List[Dict[str, Any]],
        query_type: str,
        start_time: float
    ) -> KGQueryResult:
        """Process raw query results."""
        entities = []
        relations = []
        paths = []
        
        for result in raw_results:
            # Extract entities
            for key, value in result.items():
                if isinstance(value, dict) and "name" in value and "type" in value:
                    entity = self._convert_to_entity(value, result.get(f"{key}_id"))
                    if entity not in entities:
                        entities.append(entity)
        
        return KGQueryResult(
            entities=entities,
            relations=relations,
            paths=paths,
            total_count=len(entities) + len(relations) + len(paths),
            query_time=time.time() - start_time,
            query_info={"query_type": query_type}
        )
    
    def _convert_to_entity(self, entity_data: Dict[str, Any], entity_id: str = None) -> KGEntity:
        """Convert raw entity data to KGEntity."""
        from ..ontology.schemas import EntityProperties
        
        properties_data = {
            key: value for key, value in entity_data.items()
            if key not in ["name", "type", "id", "created_at", "updated_at", "confidence", "source_documents"]
        }
        
        return KGEntity(
            id=entity_id or entity_data.get("id"),
            name=entity_data.get("name", ""),
            type=entity_data.get("type", "entity"),
            properties=EntityProperties(**properties_data),
            confidence=entity_data.get("confidence", 1.0),
            source_documents=entity_data.get("source_documents", [])
        )
    
    def _generate_cache_key(self, query_type: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key for query."""
        import hashlib
        
        query_str = json.dumps({
            "query_type": query_type,
            "parameters": parameters
        }, sort_keys=True)
        
        return f"cypher_query:{hashlib.md5(query_str.encode()).hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[KGQueryResult]:
        """Get cached result."""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return KGQueryResult(**cached_data)
        except Exception as e:
            logger.warning(f"Error getting cached result: {e}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: KGQueryResult):
        """Cache query result."""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.set(cache_key, result.dict(), ttl=self.cache_ttl)
        except Exception as e:
            logger.warning(f"Error caching result: {e}")
