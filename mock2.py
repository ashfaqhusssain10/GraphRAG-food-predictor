from neo4j import GraphDatabase

driver = GraphDatabase.driver("neo4j://127.0.0.1:7687", auth=("neo4j", "Ashfaq8790"))

with driver.session() as session:
    # Create 3 simple items with all required properties
    session.run("""
        CREATE (i1:Item {name: 'Chicken Dum Biryani', category: 'Main Course', ingredients: 'rice, chicken', flavor_profile: 'spicy', texture: 'fluffy', spice_level: 7, occasion: 'dinner', complementary_items: ['Raita'], cuisine_type: 'Indian'})
        CREATE (i2:Item {name: 'Dosa', category: 'Main Course', ingredients: 'rice, lentils', flavor_profile: 'mild', texture: 'crispy', spice_level: 2, occasion: 'breakfast', complementary_items: ['Sambar'], cuisine_type: 'South Indian'})  
        CREATE (i3:Item {name: 'Raita', category: 'Accompaniment', ingredients: 'yogurt, cucumber', flavor_profile: 'cool', texture: 'creamy', spice_level: 1, occasion: 'with meals', complementary_items: ['Biryani'], cuisine_type: 'Indian'})
        CREATE (i1)-[:CO_OCCURS {rate: 0.85, count: 127}]->(i3)
        CREATE (i2)-[:CO_OCCURS {rate: 0.90, count: 156}]->(i1)
    """)
    print("✅ Database populated!")

driver.close()