# ===================================================
# BIRYANI ITEM DIAGNOSTIC SCRIPT
# Save as: biryani_diagnostic.py
# ===================================================

from neo4j import GraphDatabase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection details
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Ashfaq8790"

def investigate_biryani_data():
    """
    Comprehensive investigation of what biryani items exist in your Neo4j database
    """
    print("🔍 BIRYANI ITEM INVESTIGATION")
    print("="*60)
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            
            # 1. Find all items containing "biryani" (case insensitive)
            print("1. SEARCHING FOR BIRYANI ITEMS:")
            print("-" * 40)
            
            biryani_query = """
            MATCH (i:Item)
            WHERE toLower(i.name) CONTAINS 'biryani'
            RETURN i.name as name, i.category as category, i.cuisine_type as cuisine_type
            ORDER BY i.name
            """
            
            result = session.run(biryani_query)
            biryani_items = []
            
            for record in result:
                item_name = record['name']
                category = record['category']
                cuisine = record['cuisine_type']
                biryani_items.append(item_name)
                print(f"   ✅ {item_name} ({category}, {cuisine})")
            
            if not biryani_items:
                print("   ❌ No items containing 'biryani' found!")
                
                # Check for similar items
                print("\n2. SEARCHING FOR SIMILAR ITEMS (mutton, dum, rice):")
                print("-" * 50)
                
                similar_query = """
                MATCH (i:Item)
                WHERE toLower(i.name) CONTAINS 'mutton' 
                   OR toLower(i.name) CONTAINS 'dum'
                   OR toLower(i.name) CONTAINS 'rice'
                RETURN i.name as name, i.category as category
                ORDER BY i.name
                LIMIT 10
                """
                
                result = session.run(similar_query)
                for record in result:
                    print(f"   📋 {record['name']} ({record['category']})")
            
            else:
                # 2. Check relationships for found biryani items
                print(f"\n2. CHECKING RELATIONSHIPS FOR {len(biryani_items)} BIRYANI ITEMS:")
                print("-" * 60)
                
                for item_name in biryani_items:
                    print(f"\n   🔍 Analyzing: {item_name}")
                    
                    # Check CO_OCCURS relationships
                    co_occurs_query = """
                    MATCH (i:Item {name: $item_name})-[co:CO_OCCURS]-(other:Item)
                    RETURN other.name as related_item, co.rate as rate, co.count as count
                    ORDER BY co.rate DESC
                    LIMIT 5
                    """
                    
                    result = session.run(co_occurs_query, {'item_name': item_name})
                    relationships = list(result)
                    
                    if relationships:
                        print(f"     ✅ Found {len(relationships)} CO_OCCURS relationships:")
                        for rel in relationships:
                            rate_pct = rel['rate'] * 100
                            print(f"       • {rel['related_item']} - {rate_pct:.1f}% ({rel['count']} times)")
                    else:
                        print(f"     ❌ No CO_OCCURS relationships found")
                    
                    # Check threshold compliance for graph tool
                    high_rate_query = """
                    MATCH (i:Item {name: $item_name})-[co:CO_OCCURS]-(other:Item)
                    WHERE co.rate >= 0.15
                    RETURN count(*) as high_rate_count
                    """
                    
                    result = session.run(high_rate_query, {'item_name': item_name})
                    high_rate_count = result.single()['high_rate_count']
                    
                    if high_rate_count > 0:
                        print(f"     ✅ {high_rate_count} relationships meet graph tool threshold (≥15%)")
                    else:
                        print(f"     ⚠️  No relationships meet graph tool threshold (≥15%)")
                        
                        # Check lower thresholds
                        lower_threshold_query = """
                        MATCH (i:Item {name: $item_name})-[co:CO_OCCURS]-(other:Item)
                        WHERE co.rate >= 0.05
                        RETURN count(*) as count_5pct
                        """
                        
                        result = session.run(lower_threshold_query, {'item_name': item_name})
                        count_5pct = result.single()['count_5pct']
                        
                        if count_5pct > 0:
                            print(f"       💡 {count_5pct} relationships available at ≥5% threshold")
            
            # 3. Check total items in database for context
            print(f"\n3. DATABASE OVERVIEW:")
            print("-" * 30)
            
            total_items_query = "MATCH (i:Item) RETURN count(i) as total"
            result = session.run(total_items_query)
            total_items = result.single()['total']
            print(f"   📊 Total items in database: {total_items}")
            
            # Sample some items
            sample_query = """
            MATCH (i:Item)
            RETURN i.name as name
            ORDER BY i.name
            LIMIT 10
            """
            
            result = session.run(sample_query)
            print(f"   📋 Sample items:")
            for record in result:
                print(f"     • {record['name']}")
                
    except Exception as e:
        print(f"❌ Error during investigation: {e}")
        
    finally:
        driver.close()

def test_alternative_queries():
    """
    Test the system with items that definitely exist in the database
    """
    print(f"\n🧪 TESTING WITH KNOWN ITEMS")
    print("="*40)
    
    # Import your processor
    from EX2 import LangChainGraphRAGProcessor
    
    processor = LangChainGraphRAGProcessor()
    
    # Test with items we know exist (based on previous success)
    test_items = [
        "Chicken Biryani",
        "Dosa", 
        "Dal Tadka",
        "Chicken Dum Biryani",  # Try this variation
        "Rice"  # Simple item that likely exists
    ]
    
    processor.setup_system()
    
    for item in test_items:
        print(f"\n🔍 Testing: '{item}'")
        query = f"What goes with {item}?"
        
        try:
            response = processor.query(query)
            if "no recommendations" in response.lower() or "need more information" in response.lower():
                print(f"   ❌ No recommendations found for '{item}'")
            else:
                print(f"   ✅ Found recommendations for '{item}'!")
                print(f"   📝 Response: {response[:100]}...")
        except Exception as e:
            print(f"   ❌ Error testing '{item}': {e}")
    
    processor.close()

def suggest_fixes():
    """
    Provide specific recommendations based on findings
    """
    print(f"\n🛠️  RECOMMENDED FIXES")
    print("="*40)
    
    print("1. Lower the co-occurrence threshold in GraphRecommendationTool:")
    print("   Change: co.rate >= 0.15")
    print("   To:     co.rate >= 0.02  (or even 0.01)")
    print()
    
    print("2. Add fuzzy name matching in extract_item_name():")
    print("   • Handle variations like 'Mutton Dum Biryani' vs 'Chicken Dum Biryani'")
    print("   • Add common synonyms (Biryani = Rice dish)")
    print()
    
    print("3. Improve semantic search by:")
    print("   • Using more descriptive queries")
    print("   • Adjusting similarity thresholds")
    print("   • Including ingredient-based matches")
    print()
    
    print("4. Add fallback recommendations:")
    print("   • If exact item not found, suggest similar category items")
    print("   • Use 'contains' matching instead of exact matching")

if __name__ == "__main__":
    print("🚀 BIRYANI ITEM DIAGNOSTIC TOOL")
    print("="*50)
    
    # Run comprehensive investigation
    investigate_biryani_data()
    
    # Test with known items
    test_alternative_queries()
    
    # Provide fix recommendations
    suggest_fixes()
    
    print(f"\n✅ DIAGNOSTIC COMPLETE!")
    print("Run this script to understand what's in your database")