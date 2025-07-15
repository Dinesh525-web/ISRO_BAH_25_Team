"""
Neo4j client for knowledge graph operations.
"""
import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import Neo4jError, ServiceUnavailable, TransientError

from app.core.config import settings
from app.core.exceptions import KnowledgeGraphException
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    """Neo4j client for knowledge graph operations."""
    
    def __init__(self):
        self.uri = settings.NEO4J_URI
        self.username = settings.NEO4J_USERNAME
        self.password = settings.NEO4J_PASSWORD
        self.database = settings.NEO4J_DATABASE
        self._driver: Optional[AsyncDriver] = None
    
    async def get_driver(self) -> AsyncDriver:
        """Get Neo4j driver instance."""
        if not self._driver:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                database=self.database,
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
                encrypted=False,  # Set to True for production
            )
        return self._driver
    
    async def close(self):
        """Close Neo4j driver."""
        if self._driver:
            await self._driver.close()
            self._driver = None
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        fetch_all: bool = True
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute a Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            fetch_all: Whether to fetch all results
            
        Returns:
            Query results
        """
        try:
            driver = await self.get_driver()
            
            async with driver.session() as session:
                result = await session.run(query, parameters or {})
                
                if fetch_all:
                    records = await result.data()
                    return records
                else:
                    record = await result.single()
                    return record.data() if record else {}
                    
        except Neo4jError as e:
            logger.error(f"Neo4j error executing query: {e}")
            raise KnowledgeGraphException(f"Neo4j error: {str(e)}")
        except Exception as e:
            logger.error(f"Error executing Neo4j query: {e}")
            raise KnowledgeGraphException(f"Query execution failed: {str(e)}")
    
    async def create_node(
        self,
        label: str,
        properties: Dict[str, Any],
        unique_key: Optional[str] = None
    ) -> str:
        """
        Create a node in the knowledge graph.
        
        Args:
            label: Node label
            properties: Node properties
            unique_key: Unique property key for MERGE operation
            
        Returns:
            Node ID
        """
        try:
            # Add metadata
            properties['created_at'] = datetime.utcnow().isoformat()
            properties['updated_at'] = datetime.utcnow().isoformat()
            
            if unique_key and unique_key in properties:
                # Use MERGE for unique nodes
                query = f"""
                MERGE (n:{label} {{{unique_key}: $unique_value}})
                SET n += $properties
                RETURN elementId(n) as id
                """
                parameters = {
                    'unique_value': properties[unique_key],
                    'properties': properties
                }
            else:
                # Use CREATE for non-unique nodes
                query = f"""
                CREATE (n:{label} $properties)
                RETURN elementId(n) as id
                """
                parameters = {'properties': properties}
            
            result = await self.execute_query(query, parameters, fetch_all=False)
            node_id = result.get('id')
            
            logger.debug(f"Created {label} node with ID: {node_id}")
            return node_id
            
        except Exception as e:
            logger.error(f"Error creating node: {e}")
            raise KnowledgeGraphException(f"Failed to create node: {str(e)}")
    
    async def get_node(
        self,
        label: str,
        property_key: str,
        property_value: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Get a node by property value.
        
        Args:
            label: Node label
            property_key: Property key to search by
            property_value: Property value to match
            
        Returns:
            Node data or None
        """
        try:
            query = f"""
            MATCH (n:{label} {{{property_key}: $value}})
            RETURN n, elementId(n) as id
            """
            
            result = await self.execute_query(
                query,
                {'value': property_value},
                fetch_all=False
            )
            
            if result:
                node_data = result['n']
                node_data['id'] = result['id']
                return node_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting node: {e}")
            return None
    
    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """
        Update node properties.
        
        Args:
            node_id: Node ID
            properties: Properties to update
            
        Returns:
            True if successful
        """
        try:
            # Add update timestamp
            properties['updated_at'] = datetime.utcnow().isoformat()
            
            query = """
            MATCH (n) WHERE elementId(n) = $node_id
            SET n += $properties
            RETURN n
            """
            
            result = await self.execute_query(
                query,
                {'node_id': node_id, 'properties': properties},
                fetch_all=False
            )
            
            success = bool(result)
            if success:
                logger.debug(f"Updated node: {node_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating node: {e}")
            return False
    
    async def delete_node(self, node_id: str) -> bool:
        """
        Delete a node and its relationships.
        
        Args:
            node_id: Node ID
            
        Returns:
            True if successful
        """
        try:
            query = """
            MATCH (n) WHERE elementId(n) = $node_id
            DETACH DELETE n
            """
            
            await self.execute_query(query, {'node_id': node_id})
            
            logger.debug(f"Deleted node: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting node: {e}")
            return False
    
    async def create_relationship(
        self,
        source_node_id: str,
        target_node_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a relationship between two nodes.
        
        Args:
            source_node_id: Source node ID
            target_node_id: Target node ID
            relationship_type: Relationship type
            properties: Relationship properties
            
        Returns:
            Relationship ID
        """
        try:
            properties = properties or {}
            properties['created_at'] = datetime.utcnow().isoformat()
            
            query = f"""
            MATCH (a) WHERE elementId(a) = $source_id
            MATCH (b) WHERE elementId(b) = $target_id
            CREATE (a)-[r:{relationship_type} $properties]->(b)
            RETURN elementId(r) as id
            """
            
            result = await self.execute_query(
                query,
                {
                    'source_id': source_node_id,
                    'target_id': target_node_id,
                    'properties': properties
                },
                fetch_all=False
            )
            
            relationship_id = result.get('id')
            logger.debug(f"Created relationship: {relationship_id}")
            return relationship_id
            
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            raise KnowledgeGraphException(f"Failed to create relationship: {str(e)}")
    
    async def get_node_relationships(
        self,
        node_id: str,
        direction: str = "both",
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for a node.
        
        Args:
            node_id: Node ID
            direction: Relationship direction ("incoming", "outgoing", "both")
            relationship_types: Filter by relationship types
            
        Returns:
            List of relationships
        """
        try:
            # Build relationship pattern based on direction
            if direction == "incoming":
                pattern = "(other)-[r]->(n)"
            elif direction == "outgoing":
                pattern = "(n)-[r]->(other)"
            else:
                pattern = "(n)-[r]-(other)"
            
            # Build type filter
            type_filter = ""
            if relationship_types:
                type_filter = ":" + "|".join(relationship_types)
            
            query = f"""
            MATCH (n) WHERE elementId(n) = $node_id
            MATCH {pattern.replace('[r]', f'[r{type_filter}]')}
            RETURN r, type(r) as relationship_type, other, elementId(other) as other_id
            """
            
            results = await self.execute_query(query, {'node_id': node_id})
            
            relationships = []
            for result in results:
                relationship_data = {
                    'relationship': result['r'],
                    'type': result['relationship_type'],
                    'related_node': result['other'],
                    'related_node_id': result['other_id'],
                }
                relationships.append(relationship_data)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error getting node relationships: {e}")
            return []
    
    async def find_path(
        self,
        source_node_id: str,
        target_node_id: str,
        max_depth: int = 5,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find path between two nodes.
        
        Args:
            source_node_id: Source node ID
            target_node_id: Target node ID
            max_depth: Maximum path depth
            relationship_types: Filter by relationship types
            
        Returns:
            List of paths
        """
        try:
            # Build relationship type filter
            type_filter = ""
            if relationship_types:
                type_filter = ":" + "|".join(relationship_types)
            
            query = f"""
            MATCH (source) WHERE elementId(source) = $source_id
            MATCH (target) WHERE elementId(target) = $target_id
            MATCH path = (source)-[r{type_filter}*1..{max_depth}]-(target)
            RETURN path, length(path) as path_length
            ORDER BY path_length
            LIMIT 10
            """
            
            results = await self.execute_query(
                query,
                {'source_id': source_node_id, 'target_id': target_node_id}
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding path: {e}")
            return []
    
    async def search_nodes(
        self,
        label: str,
        search_properties: Dict[str, Any],
        fuzzy_search: bool = False,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for nodes by properties.
        
        Args:
            label: Node label
            search_properties: Properties to search by
            fuzzy_search: Enable fuzzy text search
            limit: Maximum results
            
        Returns:
            List of matching nodes
        """
        try:
            where_clauses = []
            parameters = {}
            
            for key, value in search_properties.items():
                if fuzzy_search and isinstance(value, str):
                    where_clauses.append(f"n.{key} CONTAINS $search_{key}")
                    parameters[f"search_{key}"] = value
                else:
                    where_clauses.append(f"n.{key} = $search_{key}")
                    parameters[f"search_{key}"] = value
            
            where_clause = " AND ".join(where_clauses)
            
            query = f"""
            MATCH (n:{label})
            WHERE {where_clause}
            RETURN n, elementId(n) as id
            LIMIT {limit}
            """
            
            results = await self.execute_query(query, parameters)
            
            nodes = []
            for result in results:
                node_data = result['n']
                node_data['id'] = result['id']
                nodes.append(node_data)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error searching nodes: {e}")
            return []
    
    async def get_node_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        try:
            stats = {}
            
            # Count nodes by label
            node_query = """
            MATCH (n)
            RETURN labels(n) as labels, count(n) as count
            """
            
            node_results = await self.execute_query(node_query)
            
            label_counts = {}
            total_nodes = 0
            for result in node_results:
                labels = result['labels']
                count = result['count']
                total_nodes += count
                
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + count
            
            stats['total_nodes'] = total_nodes
            stats['nodes_by_label'] = label_counts
            
            # Count relationships by type
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) as relationship_type, count(r) as count
            """
            
            rel_results = await self.execute_query(rel_query)
            
            relationship_counts = {}
            total_relationships = 0
            for result in rel_results:
                rel_type = result['relationship_type']
                count = result['count']
                relationship_counts[rel_type] = count
                total_relationships += count
            
            stats['total_relationships'] = total_relationships
            stats['relationships_by_type'] = relationship_counts
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting node statistics: {e}")
            return {}
    
    async def create_index(self, label: str, property_key: str) -> bool:
        """
        Create an index on a node property.
        
        Args:
            label: Node label
            property_key: Property to index
            
        Returns:
            True if successful
        """
        try:
            query = f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{property_key})"
            await self.execute_query(query)
            
            logger.info(f"Created index on {label}.{property_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False
    
    async def create_constraint(
        self,
        label: str,
        property_key: str,
        constraint_type: str = "UNIQUE"
    ) -> bool:
        """
        Create a constraint on a node property.
        
        Args:
            label: Node label
            property_key: Property to constrain
            constraint_type: Constraint type
            
        Returns:
            True if successful
        """
        try:
            constraint_name = f"{label}_{property_key}_{constraint_type.lower()}"
            
            if constraint_type == "UNIQUE":
                query = f"""
                CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                FOR (n:{label}) REQUIRE n.{property_key} IS UNIQUE
                """
            else:
                query = f"""
                CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                FOR (n:{label}) REQUIRE n.{property_key} IS NOT NULL
                """
            
            await self.execute_query(query)
            
            logger.info(f"Created {constraint_type} constraint on {label}.{property_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating constraint: {e}")
            return False
    
    async def bulk_import_nodes(
        self,
        label: str,
        nodes: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> int:
        """
        Bulk import nodes.
        
        Args:
            label: Node label
            nodes: List of node data
            batch_size: Batch size for import
            
        Returns:
            Number of imported nodes
        """
        try:
            imported_count = 0
            
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                
                query = f"""
                UNWIND $nodes as node
                CREATE (n:{label})
                SET n = node
                """
                
                await self.execute_query(query, {'nodes': batch})
                imported_count += len(batch)
                
                logger.debug(f"Imported batch {i//batch_size + 1}, total: {imported_count}")
            
            logger.info(f"Bulk imported {imported_count} {label} nodes")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error bulk importing nodes: {e}")
            return 0
    
    async def health_check(self) -> bool:
        """Check Neo4j health."""
        try:
            query = "RETURN 1 as health"
            result = await self.execute_query(query, fetch_all=False)
            return result.get('health') == 1
            
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False
    
    async def clear_database(self) -> bool:
        """Clear all data from database (use with caution)."""
        try:
            query = "MATCH (n) DETACH DELETE n"
            await self.execute_query(query)
            
            logger.warning("Cleared Neo4j database")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False
