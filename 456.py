"""
Quick fix to complete the enhancement that failed due to type comparison error
Run this after the database builder to complete the multi-dimensional enhancement
"""

import logging
from neo4j import GraphDatabase
from typing import Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Ashfaq8790"

def calculate_event_suitability_fixed(item: Dict, event_name: str) -> Tuple[float, str]:
    """Fixed version of event suitability calculation"""
    score = 0.5
    reasons = []
    
    # Get spice level safely
    spice_level = item.get('spice_level')
    if not isinstance(spice_level, (int, float)):
        spice_level = 5  # Default
    
    # Corporate events need professional-friendly foods
    if 'Corporate' in event_name:
        if spice_level <= 4:
            score += 0.3
            reasons.append("low spice appropriate for business")
        else:
            score -= 0.2
            reasons.append("high spice may not suit business context")
        
        if item.get('category') in ['Main Course', 'Snacks']:
            score += 0.2
            reasons.append("suitable category for corporate dining")
    
    # Wedding events are more permissive and festive
    elif 'Wedding' in event_name or 'Engagement' in event_name:
        if item.get('category') in ['Main Course', 'Desserts', 'Snacks']:
            score += 0.3
            reasons.append("festive food appropriate for celebrations")
        
        if spice_level >= 6:
            score += 0.1
            reasons.append("flavorful food suits celebration")
    
    # Family gatherings are very permissive
    elif 'Family' in event_name or 'House' in event_name:
        score += 0.2
        reasons.append("suitable for family contexts")
    
    # Party events favor exciting foods
    elif 'Party' in event_name:
        if item.get('category') in ['Snacks', 'Beverages', 'Desserts']:
            score += 0.3
            reasons.append("party-friendly food category")
    
    score = max(0.0, min(1.0, score))
    reasoning = "; ".join(reasons) if reasons else "general suitability"
    
    return score, reasoning

def calculate_meal_appropriateness_fixed(item: Dict, meal_name: str) -> Tuple[float, str]:
    """Fixed version of meal appropriateness calculation"""
    score = 0.3
    reasons = []
    
    category = item.get('category', '').lower()
    spice_level = item.get('spice_level')
    if not isinstance(spice_level, (int, float)):
        spice_level = 5  # Default
    
    if meal_name == 'Breakfast':
        if 'beverage' in category or 'light' in category:
            score += 0.4
            reasons.append("light foods appropriate for morning")
        
        if spice_level <= 3:
            score += 0.3
            reasons.append("mild spice suits morning palate")
        else:
            score -= 0.2
            reasons.append("high spice less suitable for morning")
    
    elif meal_name == 'Lunch':
        if 'main course' in category:
            score += 0.4
            reasons.append("substantial food for midday meal")
        
        if 2 <= spice_level <= 7:
            score += 0.2
            reasons.append("moderate spice appropriate for lunch")
    
    elif meal_name == 'Dinner':
        if 'main course' in category or 'dessert' in category:
            score += 0.4
            reasons.append("elaborate food suitable for evening")
        
        score += 0.1
        reasons.append("evening meal allows variety")
    
    elif meal_name == 'Snacks':
        if 'snack' in category or 'appetizer' in category:
            score += 0.5
            reasons.append("designed for snacking")
        
        if 'beverage' in category:
            score += 0.3
            reasons.append("beverages complement snacks")
    
    score = max(0.0, min(1.0, score))
    reasoning = "; ".join(reasons) if reasons else "general meal appropriateness"
    
    return score, reasoning

def complete_enhancement():
    """Complete the enhancement that failed"""
    logger.info("Completing enhancement with fixed type handling...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Get unique event types
            event_result = session.run("MATCH (e:Event) RETURN e.name as name")
            event_types = [record['name'] for record in event_result]
            
            # Get unique meal types  
            meal_result = session.run("MATCH (m:Meal) RETURN m.name as name")
            meal_types = [record['name'] for record in meal_result]
            
            # Get all items with proper numeric spice levels
            items_result = session.run("""
                MATCH (i:Item)
                RETURN i.name as name, i.spice_level_numeric as spice_level, 
                       i.category as category, i.cuisine_type as cuisine_type
            """)
            items = [record.data() for record in items_result]
            
            logger.info(f"Found {len(items)} items, {len(event_types)} events, {len(meal_types)} meals")
            
            # Create SUITABLE_FOR relationships
            logger.info("Creating SUITABLE_FOR relationships...")
            suitability_query = """
            MATCH (i:Item {name: $item_name})
            MATCH (e:Event {name: $event_name})
            CREATE (i)-[:SUITABLE_FOR {
                suitability_score: $score,
                reasoning: $reasoning,
                derived_from: 'business_logic'
            }]->(e)
            """
            
            suitable_count = 0
            for item in items:
                for event_name in event_types:
                    score, reasoning = calculate_event_suitability_fixed(item, event_name)
                    
                    if score >= 0.5:
                        try:
                            session.run(suitability_query, {
                                'item_name': item['name'],
                                'event_name': event_name,
                                'score': score,
                                'reasoning': reasoning
                            })
                            suitable_count += 1
                        except Exception as e:
                            logger.debug(f"Failed SUITABLE_FOR: {item['name']} -> {event_name}: {e}")
            
            logger.info(f"✓ Created {suitable_count} SUITABLE_FOR relationships")
            
            # Create SERVED_AT relationships
            logger.info("Creating SERVED_AT relationships...")
            served_at_query = """
            MATCH (i:Item {name: $item_name})
            MATCH (m:Meal {name: $meal_name})
            CREATE (i)-[:SERVED_AT {
                appropriateness_score: $score,
                cultural_reasoning: $reasoning,
                derived_from: 'cultural_knowledge'
            }]->(m)
            """
            
            served_count = 0
            for item in items:
                for meal_name in meal_types:
                    score, reasoning = calculate_meal_appropriateness_fixed(item, meal_name)
                    
                    if score >= 0.4:
                        try:
                            session.run(served_at_query, {
                                'item_name': item['name'],
                                'meal_name': meal_name,
                                'score': score,
                                'reasoning': reasoning
                            })
                            served_count += 1
                        except Exception as e:
                            logger.debug(f"Failed SERVED_AT: {item['name']} -> {meal_name}: {e}")
            
            logger.info(f"✓ Created {served_count} SERVED_AT relationships")
            
            # Final validation
            logger.info("Final validation...")
            validation_queries = {
                'suitable_for_rels': "MATCH ()-[r:SUITABLE_FOR]->() RETURN count(r) as count",
                'served_at_rels': "MATCH ()-[r:SERVED_AT]->() RETURN count(r) as count",
                'contains_rels': "MATCH ()-[r:CONTAINS]->() RETURN count(r) as count"
            }
            
            for query_name, query in validation_queries.items():
                result = session.run(query)
                count = result.single()['count']
                logger.info(f"  {query_name}: {count}")
            
            logger.info("✓ Enhancement completed successfully!")
            
    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    complete_enhancement()