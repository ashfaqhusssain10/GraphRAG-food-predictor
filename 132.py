# ================================================
# COMPLETE CHROMADB + NEO4J DIAGNOSTIC TOOL
# Save this as: diagnostic_complete.py
# ================================================

import os
import sys
import socket
import sqlite3
import subprocess
from pathlib import Path

def check_chromadb_issues():
    """Comprehensive ChromaDB diagnostic"""
    print("🔍 CHROMADB DIAGNOSTICS")
    print("="*50)
    
    # Check 1: ChromaDB installation
    try:
        import chromadb
        print(f"✅ ChromaDB installed: {chromadb.__version__}")
    except ImportError as e:
        print(f"❌ ChromaDB not installed: {e}")
        print("   Fix: pip install chromadb")
        return False
    
    # Check 2: Directory permissions
    chroma_dirs = ["./chroma_db", "./chroma", "chroma_db", "chroma"]
    for dir_path in chroma_dirs:
        if os.path.exists(dir_path):
            print(f"📁 Found ChromaDB directory: {dir_path}")
            try:
                # Test write permissions
                test_file = os.path.join(dir_path, "test_write.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                print(f"✅ Directory {dir_path} is writable")
            except PermissionError:
                print(f"❌ Permission denied for {dir_path}")
                print(f"   Fix: chmod 755 {dir_path}")
                return False
            except Exception as e:
                print(f"❌ Directory issue: {e}")
    
    # Check 3: Port conflicts
    ports_to_check = [8000, 8001, 8080]
    for port in ports_to_check:
        if is_port_in_use("127.0.0.1", port):
            print(f"⚠️  Port {port} is in use (potential ChromaDB conflict)")
        else:
            print(f"✅ Port {port} is available")
    
    # Check 4: SQLite conflicts
    sqlite_files = list(Path(".").rglob("*.sqlite*")) + list(Path(".").rglob("*.db"))
    if sqlite_files:
        print(f"📊 Found {len(sqlite_files)} database files:")
        for db_file in sqlite_files[:5]:  # Show first 5
            try:
                conn = sqlite3.connect(str(db_file), timeout=1)
                conn.close()
                print(f"✅ {db_file} - accessible")
            except sqlite3.OperationalError:
                print(f"❌ {db_file} - LOCKED or corrupted")
                print(f"   Fix: Delete or move {db_file}")
                return False
    
    # Check 5: Test ChromaDB creation
    try:
        print("🧪 Testing ChromaDB initialization...")
        client = chromadb.Client()
        collection = client.get_or_create_collection("test_collection")
        collection.add(
            documents=["test document"],
            ids=["test_id"]
        )
        print("✅ ChromaDB test successful")
        return True
    except Exception as e:
        print(f"❌ ChromaDB initialization failed: {e}")
        print("   This is likely your main issue!")
        return False

def is_port_in_use(host, port):
    """Check if a port is in use"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def check_neo4j_connection():
    """Test Neo4j connection separately"""
    print("\n🔍 NEO4J DIAGNOSTICS")
    print("="*50)
    
    # Check 1: Port accessibility
    if is_port_in_use("127.0.0.1", 7687):
        print("✅ Neo4j port 7687 is accessible")
    else:
        print("❌ Neo4j port 7687 is NOT accessible")
        print("   Fix: Start Neo4j Desktop")
        return False
    
    # Check 2: neo4j driver
    try:
        from neo4j import GraphDatabase
        print("✅ Neo4j driver installed")
    except ImportError:
        print("❌ Neo4j driver not installed")
        print("   Fix: pip install neo4j")
        return False
    
    # Check 3: Connection test
    try:
        driver = GraphDatabase.driver(
            "bolt://127.0.0.1:7687", 
            auth=("neo4j", "Ashfaq8790")
        )
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_val = result.single()["test"]
            if test_val == 1:
                print("✅ Neo4j connection successful")
                driver.close()
                return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

def check_system_resources():
    """Check system resources that might affect both databases"""
    print("\n🔍 SYSTEM DIAGNOSTICS")
    print("="*50)
    
    # Check available disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (1024**3)
        print(f"💾 Free disk space: {free_gb} GB")
        if free_gb < 1:
            print("⚠️  Low disk space might cause database issues")
    except:
        pass
    
    # Check running processes
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'neo4j' in result.stdout:
            print("✅ Neo4j process is running")
        else:
            print("❌ Neo4j process not found")
            
        if 'chroma' in result.stdout:
            print("⚠️  ChromaDB process already running (might conflict)")
    except:
        print("ℹ️  Could not check running processes (Windows?)")

def suggest_fixes():
    """Provide specific fix suggestions"""
    print("\n🛠️  SUGGESTED FIXES")
    print("="*50)
    print("1. Clean ChromaDB cache:")
    print("   rm -rf ./chroma_db ./chroma *.sqlite *.db")
    print()
    print("2. Reinstall ChromaDB:")
    print("   pip uninstall chromadb")
    print("   pip install chromadb")
    print()
    print("3. Use temporary directory for ChromaDB:")
    print("   import tempfile")
    print("   chroma_client = chromadb.Client(Settings(")
    print("       persist_directory=tempfile.mkdtemp()")
    print("   ))")
    print()
    print("4. Check for conflicting processes:")
    print("   kill any existing ChromaDB/Neo4j processes")
    print()
    print("5. Try alternative ChromaDB backend:")
    print("   Use in-memory client instead of persistent")

def main():
    print("🚀 COMPLETE GRAPHRAG DIAGNOSTIC")
    print("="*60)
    
    chromadb_ok = check_chromadb_issues()
    neo4j_ok = check_neo4j_connection()
    check_system_resources()
    
    print("\n📊 DIAGNOSTIC SUMMARY")
    print("="*50)
    print(f"ChromaDB Status: {'✅ OK' if chromadb_ok else '❌ FAILED'}")
    print(f"Neo4j Status:    {'✅ OK' if neo4j_ok else '❌ FAILED'}")
    
    if not chromadb_ok or not neo4j_ok:
        suggest_fixes()
    else:
        print("🎉 Both databases appear to be working!")
        print("   The issue might be in your specific code implementation.")

if __name__ == "__main__":
    main()