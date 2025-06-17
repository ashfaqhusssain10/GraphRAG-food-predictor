"""
Complete Database Builder for GraphRAG System
Builds Neo4j database from DF1.xlsx and then runs the enhanced GraphRAG system
"""

import os
import logging
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter

# Import your existing GraphRAG components (will be imported later after database is built)
# from EX2 import LangChainGraphRAGProcessor, enhance_existing_knowledge_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Ashfaq8790"

class DatabaseBuilder:
    """
    Builds complete Neo4j database from DF1.xlsx
    Extracts rich item information and creates relationships
    """
    
    def __init__(self):
        logger.info("Initializing Database Builder...")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.df1_data = None
        self.unique_items = {}  # item_name -> consolidated properties
        self.menu_compositions = {}  # menu_id -> list of items
        self.co_occurrence_stats = defaultdict(lambda: defaultdict(int))
        
    def load_df1_data(self, file_path: str = "DF1.xlsx") -> pd.DataFrame:
        """Load and validate DF1.xlsx data"""
        logger.info(f"Loading data from {file_path}...")
        
        try:
            self.df1_data = pd.read_excel(file_path, sheet_name='Sheet1')
            logger.info(f"✓ Loaded {len(self.df1_data)} records from DF1.xlsx")
            
            # Validate required columns
            required_cols = ['menu_id', 'Event Type', 'Meal Type', 'Category', 'item_name', 'Item Description']
            missing_cols = [col for col in required_cols if col not in self.df1_data.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Clean data
            self.df1_data = self.df1_data.dropna(subset=['item_name', 'Item Description'])
            logger.info(f"✓ After cleaning: {len(self.df1_data)} valid records")
            
            return self.df1_data
            
        except Exception as e:
            logger.error(f"Error loading DF1 data: {e}")
            raise
    
    def parse_item_description(self, description: str) -> Dict:
        """
        Parse the rich Item Description field to extract structured properties
        
        Example description:
        "Ingredients: Apricots, cream, sugar, gelatin or agar, nuts 
         Preparation: Chilled dessert with apricot puree and cream 
         Flavor Profile: Sweet, fruity, creamy, refreshing 
         Spice Level: None (0/10) 
         Dietary: Vegetarian, contains dairy 
         Classification: Modern fusion dessert 
         Texture: Smooth, creamy, light 
         Complementary Items: Cookies, light snacks 
         Occasion: Summer dessert, elegant dining"
        """
        properties = {}
        
        # Define extraction patterns
        patterns = {
            'ingredients': r'Ingredients:\s*([^\n]+)',
            'preparation': r'Preparation:\s*([^\n]+)',
            'flavor_profile': r'Flavor Profile:\s*([^\n]+)',
            'spice_level': r'Spice Level:\s*([^\n]+)',
            'dietary': r'Dietary:\s*([^\n]+)',
            'classification': r'Classification:\s*([^\n]+)',
            'texture': r'Texture:\s*([^\n]+)',
            'complementary_items': r'Complementary Items:\s*([^\n]+)',
            'occasion': r'Occasion:\s*([^\n]+)'
        }
        
        # Extract each property
        for prop_name, pattern in patterns.items():
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                properties[prop_name] = match.group(1).strip()
        
        # Extract numeric spice level
        spice_level = properties.get('spice_level', '')
        spice_match = re.search(r'\((\d+)/10\)', spice_level)
        if spice_match:
            properties['spice_level_numeric'] = int(spice_match.group(1))
        else:
            # Default based on description
            if 'none' in spice_level.lower() or 'no spice' in spice_level.lower():
                properties['spice_level_numeric'] = 0
            elif 'mild' in spice_level.lower():
                properties['spice_level_numeric'] = 3
            elif 'medium' in spice_level.lower():
                properties['spice_level_numeric'] = 5
            elif 'hot' in spice_level.lower() or 'high' in spice_level.lower():
                properties['spice_level_numeric'] = 8
            else:
                properties['spice_level_numeric'] = 5  # Default medium
        
        # Parse complementary items into a list
        complementary = properties.get('complementary_items', '')
        if complementary:
            # Split by commas and clean up
            comp_list = [item.strip() for item in complementary.split(',')]
            properties['complementary_items_list'] = comp_list
        
        return properties
    
    def consolidate_item_data(self):
        """
        Consolidate item data across all menu appearances
        Same item might appear in multiple menus with same/similar descriptions
        """
        logger.info("Consolidating item data across menus...")
        
        for _, row in self.df1_data.iterrows():
            item_name = row['item_name']
            category = row['Category']
            description = row['Item Description']
            
            # Parse the rich description
            parsed_props = self.parse_item_description(description)
            
            if item_name not in self.unique_items:
                # First occurrence of this item
                self.unique_items[item_name] = {
                    'name': item_name,
                    'category': category,
                    'appearances': 1,
                    'event_types': set([row['Event Type']]),
                    'meal_types': set([row['Meal Type']]),
                    **parsed_props
                }
            else:
                # Item already exists, update counters and sets
                self.unique_items[item_name]['appearances'] += 1
                self.unique_items[item_name]['event_types'].add(row['Event Type'])
                self.unique_items[item_name]['meal_types'].add(row['Meal Type'])
                
                # Update properties if more detailed description is found
                for key, value in parsed_props.items():
                    if value and (key not in self.unique_items[item_name] or not self.unique_items[item_name][key]):
                        self.unique_items[item_name][key] = value
        
        # Convert sets to lists for Neo4j storage
        for item_data in self.unique_items.values():
            item_data['event_types'] = list(item_data['event_types'])
            item_data['meal_types'] = list(item_data['meal_types'])
        
        logger.info(f"✓ Consolidated {len(self.unique_items)} unique items")
    
    def analyze_menu_compositions(self):
        """Analyze which items appear together in menus for co-occurrence relationships"""
        logger.info("Analyzing menu compositions for co-occurrence patterns...")
        
        # Group by menu_id
        for menu_id, group in self.df1_data.groupby('menu_id'):
            items_in_menu = group['item_name'].tolist()
            self.menu_compositions[menu_id] = {
                'items': items_in_menu,
                'event_type': group['Event Type'].iloc[0],
                'meal_type': group['Meal Type'].iloc[0],
                'item_count': len(items_in_menu)
            }
            
            # Calculate co-occurrences for this menu
            for i, item1 in enumerate(items_in_menu):
                for j, item2 in enumerate(items_in_menu):
                    if i != j:  # Don't count self-occurrences
                        self.co_occurrence_stats[item1][item2] += 1
        
        logger.info(f"✓ Analyzed {len(self.menu_compositions)} menus")
        logger.info(f"✓ Generated co-occurrence data for {len(self.co_occurrence_stats)} items")
    
    def clear_existing_data(self):
        """Clear existing data from Neo4j to start fresh"""
        logger.info("Clearing existing Neo4j data...")
        
        with self.driver.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
        
        logger.info("✓ Cleared existing Neo4j data")
    
    def create_item_nodes(self):
        """Create Item nodes in Neo4j with rich properties"""
        logger.info("Creating Item nodes in Neo4j...")
        
        item_creation_query = """
        CREATE (i:Item {
            name: $name,
            category: $category,
            appearances: $appearances,
            ingredients: $ingredients,
            preparation: $preparation,
            flavor_profile: $flavor_profile,
            spice_level: $spice_level,
            spice_level_numeric: $spice_level_numeric,
            dietary: $dietary,
            classification: $classification,
            texture: $texture,
            complementary_items: $complementary_items,
            complementary_items_list: $complementary_items_list,
            occasion: $occasion,
            event_types: $event_types,
            meal_types: $meal_types,
            cuisine_type: $cuisine_type
        })
        """
        
        created_count = 0
        
        with self.driver.session() as session:
            for item_name, item_data in self.unique_items.items():
                try:
                    # Determine cuisine type from classification or category
                    cuisine_type = self._determine_cuisine_type(item_data)
                    
                    session.run(item_creation_query, {
                        'name': item_data['name'],
                        'category': item_data['category'],
                        'appearances': item_data['appearances'],
                        'ingredients': item_data.get('ingredients', ''),
                        'preparation': item_data.get('preparation', ''),
                        'flavor_profile': item_data.get('flavor_profile', ''),
                        'spice_level': item_data.get('spice_level', ''),
                        'spice_level_numeric': item_data.get('spice_level_numeric', 5),
                        'dietary': item_data.get('dietary', ''),
                        'classification': item_data.get('classification', ''),
                        'texture': item_data.get('texture', ''),
                        'complementary_items': item_data.get('complementary_items', ''),
                        'complementary_items_list': item_data.get('complementary_items_list', []),
                        'occasion': item_data.get('occasion', ''),
                        'event_types': item_data['event_types'],
                        'meal_types': item_data['meal_types'],
                        'cuisine_type': cuisine_type
                    })
                    created_count += 1
                    
                    if created_count % 100 == 0:
                        logger.info(f"  Created {created_count} Item nodes...")
                        
                except Exception as e:
                    logger.warning(f"Failed to create Item node for {item_name}: {e}")
        
        logger.info(f"✓ Created {created_count} Item nodes")
    
    def _determine_cuisine_type(self, item_data: Dict) -> str:
        """Determine cuisine type from item classification and characteristics"""
        classification = item_data.get('classification', '').lower()
        category = item_data.get('category', '').lower()
        name = item_data.get('name', '').lower()
        
        # Indian cuisine indicators
        indian_keywords = ['indian', 'south indian', 'north indian', 'traditional', 'biryani', 'dal', 'curry', 'masala', 'tandoor']
        if any(keyword in classification or keyword in name for keyword in indian_keywords):
            return 'Indian'
        
        # Continental/Western indicators
        western_keywords = ['continental', 'western', 'modern', 'fusion', 'pasta', 'pizza', 'sandwich']
        if any(keyword in classification for keyword in western_keywords):
            return 'Continental'
        
        # Chinese indicators
        chinese_keywords = ['chinese', 'noodles', 'manchurian', 'fried rice']
        if any(keyword in classification or keyword in name for keyword in chinese_keywords):
            return 'Chinese'
        
        # Default based on category
        if 'dessert' in category:
            return 'International'
        elif 'beverage' in category:
            return 'Universal'
        else:
            return 'Indian'  # Default for most items
    
    def create_co_occurrence_relationships(self):
        """Create CO_OCCURS relationships based on menu co-appearances"""
        logger.info("Creating CO_OCCURS relationships...")
        
        # Calculate total menu count for rate calculation
        total_menus = len(self.menu_compositions)
        
        co_occurs_query = """
        MATCH (i1:Item {name: $item1})
        MATCH (i2:Item {name: $item2})
        CREATE (i1)-[:CO_OCCURS {
            count: $count,
            rate: $rate,
            total_menus: $total_menus
        }]->(i2)
        """
        
        created_count = 0
        
        with self.driver.session() as session:
            for item1, co_items in self.co_occurrence_stats.items():
                for item2, count in co_items.items():
                    # Calculate co-occurrence rate
                    rate = count / total_menus
                    
                    # Only create relationships for meaningful co-occurrences
                    if count >= 2 and rate >= 0.01:  # At least 2 occurrences and 1% rate
                        try:
                            session.run(co_occurs_query, {
                                'item1': item1,
                                'item2': item2,
                                'count': count,
                                'rate': rate,
                                'total_menus': total_menus
                            })
                            created_count += 1
                            
                            if created_count % 1000 == 0:
                                logger.info(f"  Created {created_count} CO_OCCURS relationships...")
                                
                        except Exception as e:
                            logger.debug(f"Failed to create CO_OCCURS: {item1} -> {item2}: {e}")
        
        logger.info(f"✓ Created {created_count} CO_OCCURS relationships")
    
    def validate_database(self):
        """Validate the created database structure"""
        logger.info("Validating created database...")
        
        validation_queries = {
            'total_items': "MATCH (i:Item) RETURN count(i) as count",
            'items_with_spice_level': "MATCH (i:Item) WHERE i.spice_level_numeric IS NOT NULL RETURN count(i) as count",
            'items_with_ingredients': "MATCH (i:Item) WHERE i.ingredients <> '' RETURN count(i) as count",
            'total_co_occurs': "MATCH ()-[r:CO_OCCURS]->() RETURN count(r) as count",
            'high_co_occurs': "MATCH ()-[r:CO_OCCURS]->() WHERE r.rate >= 0.1 RETURN count(r) as count",
            'categories': "MATCH (i:Item) RETURN DISTINCT i.category as category ORDER BY category",
            'cuisine_types': "MATCH (i:Item) RETURN DISTINCT i.cuisine_type as cuisine ORDER BY cuisine"
        }
        
        with self.driver.session() as session:
            for query_name, query in validation_queries.items():
                try:
                    result = session.run(query)
                    if query_name in ['categories', 'cuisine_types']:
                        values = [record[query_name.rstrip('s')] for record in result]
                        logger.info(f"  {query_name}: {values}")
                    else:
                        count = result.single()['count']
                        logger.info(f"  {query_name}: {count}")
                except Exception as e:
                    logger.error(f"Validation query {query_name} failed: {e}")
        
        logger.info("✓ Database validation complete")
    
    def build_complete_database(self, df1_file_path: str = "DF1.xlsx"):
        """Complete database building process"""
        logger.info("="*60)
        logger.info("BUILDING COMPLETE NEO4J DATABASE FROM DF1.XLSX")
        logger.info("="*60)
        
        try:
            # Step 1: Load and parse data
            self.load_df1_data(df1_file_path)
            
            # Step 2: Consolidate item information
            self.consolidate_item_data()
            
            # Step 3: Analyze menu compositions
            self.analyze_menu_compositions()
            
            # Step 4: Clear existing data
            self.clear_existing_data()
            
            # Step 5: Create Item nodes
            self.create_item_nodes()
            
            # Step 6: Create CO_OCCURS relationships
            self.create_co_occurrence_relationships()
            
            # Step 7: Validate the database
            self.validate_database()
            
            logger.info("✓ Database building complete!")
            return True
            
        except Exception as e:
            logger.error(f"Database building failed: {e}")
            return False
    
    def close(self):
        """Clean up resources"""
        if self.driver:
            self.driver.close()

def main():
    """
    Complete pipeline: Build database from DF1.xlsx, then run GraphRAG system
    """
    print("="*80)
    print("COMPLETE GRAPHRAG SYSTEM INITIALIZATION FROM DF1.XLSX")
    print("="*80)
    
    # Step 1: Build the database from DF1.xlsx
    builder = DatabaseBuilder()
    
    success = builder.build_complete_database("DF1.xlsx")
    if not success:
        print("✗ Database building failed. Exiting.")
        builder.close()
        return
    
    builder.close()
    
    print("\n" + "="*60)
    print("INITIALIZING GRAPHRAG SYSTEM WITH POPULATED DATABASE")
    print("="*60)
    
    # Step 2: Import and initialize the GraphRAG system (now with populated database)
    try:
        from EX2 import LangChainGraphRAGProcessor, enhance_existing_knowledge_graph
        
        processor = LangChainGraphRAGProcessor()
        processor.setup_system()
        
        print("\n" + "="*60)
        print("TESTING BASIC GRAPHRAG FUNCTIONALITY")
        print("="*60)
        
        # Step 3: Test basic functionality
        test_queries = [
            "What goes with Apricot Delight?",
            "Recommend something for Badam Milk",
            "What are some novel pairings for Cut Mirchi Bajji?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            try:
                response = processor.query(query)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {e}")
        
        print("\n" + "="*60)
        print("ENHANCING WITH MULTI-DIMENSIONAL INTELLIGENCE")
        print("="*60)
        
        # Step 4: Add multi-dimensional intelligence
        try:
            enhance_existing_knowledge_graph(processor, "DF1.xlsx")
            
            print("\n" + "="*60)
            print("TESTING ENHANCED CONTEXTUAL CAPABILITIES")
            print("="*60)
            
            contextual_queries = [
                "What goes with Apricot Delight for a House Warming lunch?",
                "Recommend breakfast items that pair well with Badam Milk",
                "Suggest snack items that go with Cut Mirchi Bajji"
            ]
            
            for query in contextual_queries:
                print(f"\nContextual Query: {query}")
                try:
                    response = processor.query(query)
                    print(f"Enhanced Response: {response}")
                except Exception as e:
                    print(f"Error: {e}")
                    
        except Exception as e:
            print(f"Enhancement failed: {e}")
            print("System will continue with basic functionality")
        
        # Clean up
        processor.close()
        
    except ImportError as e:
        print(f"Could not import GraphRAG components: {e}")
        print("Database built successfully, but GraphRAG system test skipped.")
        print("Make sure EX2.py is available to test the full system.")
    print("\n" + "="*80)
    print("COMPLETE SYSTEM READY FOR USE")
    print("="*80)

if __name__ == "__main__":
    print("Testing data.py connection...")
    main()