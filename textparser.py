"""
Quick test script to verify DF1.xlsx parsing works correctly
Run this first to validate data extraction before building the full database
"""

import pandas as pd
import re
from typing import Dict

def parse_item_description(description: str) -> Dict:
    """Parse the rich Item Description field to extract structured properties"""
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
    
    return properties

def test_data_parsing():
    """Test the data parsing functionality"""
    print("="*60)
    print("TESTING DF1.XLSX DATA PARSING")
    print("="*60)
    
    try:
        # Load the data
        df = pd.read_excel("DF1.xlsx", sheet_name='Sheet1')
        print(f"✓ Loaded {len(df)} records from DF1.xlsx")
        
        # Show data structure
        print(f"\nColumns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        
        # Test parsing on first few items
        print("\n" + "="*40)
        print("TESTING ITEM DESCRIPTION PARSING")
        print("="*40)
        
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            item_name = row['item_name']
            description = row['Item Description']
            
            print(f"\n--- Item {i+1}: {item_name} ---")
            print(f"Category: {row['Category']}")
            print(f"Event Type: {row['Event Type']}")
            print(f"Meal Type: {row['Meal Type']}")
            
            # Parse the description
            parsed = parse_item_description(description)
            
            print("\nParsed Properties:")
            for key, value in parsed.items():
                print(f"  {key}: {value}")
        
        # Analyze data distribution
        print("\n" + "="*40)
        print("DATA DISTRIBUTION ANALYSIS")
        print("="*40)
        
        print(f"Unique items: {df['item_name'].nunique()}")
        print(f"Unique menus: {df['menu_id'].nunique()}")
        print(f"Unique categories: {df['Category'].nunique()}")
        print(f"Unique event types: {df['Event Type'].nunique()}")
        print(f"Unique meal types: {df['Meal Type'].nunique()}")
        
        print("\nTop 10 Categories:")
        print(df['Category'].value_counts().head(10))
        
        print("\nEvent Types:")
        print(df['Event Type'].value_counts())
        
        print("\nMeal Types:")
        print(df['Meal Type'].value_counts())
        
        # Test menu composition analysis
        print("\n" + "="*40)
        print("MENU COMPOSITION ANALYSIS")
        print("="*40)
        
        menu_sizes = df.groupby('menu_id').size()
        print(f"Average items per menu: {menu_sizes.mean():.1f}")
        print(f"Menu size range: {menu_sizes.min()} - {menu_sizes.max()}")
        print(f"Most common menu sizes:")
        print(menu_sizes.value_counts().head(5))
        
        # Sample menu
        sample_menu = df[df['menu_id'] == df['menu_id'].iloc[0]]
        print(f"\nSample menu ({sample_menu['menu_id'].iloc[0]}):")
        for _, item in sample_menu.iterrows():
            print(f"  - {item['item_name']} ({item['Category']})")
        
        print("\n✓ Data parsing test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_data_parsing()
    if success:
        print("\n" + "="*60)
        print("READY TO BUILD DATABASE!")
        print("Run the main database builder script next.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("FIX DATA ISSUES BEFORE PROCEEDING")
        print("="*60)