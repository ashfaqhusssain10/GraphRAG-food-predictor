"""
Neo4j Connection Test and Troubleshooting Script
This script helps diagnose Neo4j connectivity issues
"""

import sys
from neo4j import GraphDatabase
import socket

# Connection parameters from your code
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Ashfaq8790"

def test_port_connectivity():
    """Test if port 7687 is open"""
    print("Testing port connectivity...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('127.0.0.1', 7687))
        sock.close()
        
        if result == 0:
            print("✓ Port 7687 is open and accepting connections")
            return True
        else:
            print("✗ Port 7687 is not accessible")
            return False
    except Exception as e:
        print(f"✗ Port test failed: {e}")
        return False

def test_neo4j_connection():
    """Test Neo4j database connection"""
    print("\nTesting Neo4j connection...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # Test the connection
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful!' as message")
            record = result.single()
            message = record["message"]
            print(f"✓ Neo4j connection successful: {message}")
            
            # Check if there's any data
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            count = result.single()["node_count"]
            print(f"✓ Database contains {count} nodes")
            
            # Check for Item nodes specifically
            result = session.run("MATCH (i:Item) RETURN count(i) as item_count")
            item_count = result.single()["item_count"]
            print(f"✓ Database contains {item_count} Item nodes")
            
        driver.close()
        return True
        
    except Exception as e:
        print(f"✗ Neo4j connection failed: {e}")
        return False

def provide_troubleshooting_steps():
    """Provide step-by-step troubleshooting guide"""
    print("\n" + "="*60)
    print("TROUBLESHOOTING STEPS")
    print("="*60)
    
    print("\n1. START NEO4J DATABASE:")
    print("   - If using Neo4j Desktop: Open Neo4j Desktop and start your database")
    print("   - If using Neo4j Community Server: Run 'neo4j start' in terminal")
    print("   - If using Docker: Run 'docker run --publish=7474:7474 --publish=7687:7687 neo4j'")
    
    print("\n2. VERIFY NEO4J IS RUNNING:")
    print("   - Open http://localhost:7474 in your browser")
    print("   - You should see the Neo4j Browser interface")
    print("   - Try logging in with username 'neo4j' and your password")
    
    print("\n3. CHECK CONNECTION SETTINGS:")
    print("   - Default Neo4j port: 7687")
    print("   - Default username: neo4j")
    print("   - Password: Check if it's 'Ashfaq8790' or was changed")
    
    print("\n4. POPULATE DATABASE (if empty):")
    print("   - Run your data loading script to populate Item nodes")
    print("   - Ensure CO_OCCURS relationships are created")
    
    print("\n5. ALTERNATIVE SOLUTIONS:")
    print("   - Use Neo4j Aura (cloud database)")
    print("   - Use local SQLite for development")
    print("   - Mock the data for testing")

def main():
    print("Neo4j Connection Diagnostic Tool")
    print("="*40)
    
    # Test 1: Port connectivity
    port_ok = test_port_connectivity()
    
    # Test 2: Neo4j connection
    connection_ok = test_neo4j_connection()
    
    # Provide guidance based on results
    if port_ok and connection_ok:
        print("\n✓ All tests passed! Neo4j is running and accessible.")
        print("Your original script should work now.")
    else:
        provide_troubleshooting_steps()

if __name__ == "__main__":
    main()