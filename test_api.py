import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_health_check():
    print("\n🔍 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is healthy!")
            print(f"   📊 Documents in DB: {data['documents_count']}")
            print(f"   🤖 Model: {data['model']}")
            return True
        else:
            print(f"❌ Health check failed: Status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server!")
        print("   Make sure server is running: python main.py")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_document_ingestion(pdf_path):
    print(f"\n📄 Testing Document Ingestion...")
    print(f"   File: {pdf_path}")
    
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': (pdf_path, f, 'application/pdf')}
            print("   ⏳ Uploading and processing...")
            response = requests.post(f"{BASE_URL}/ingest", files=files, timeout=300)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Document ingested successfully!")
            print(f"   📊 Chunks processed: {data['chunks_processed']}")
            print(f"   📁 Filename: {data['filename']}")
            return True
        else:
            print(f"❌ Ingestion failed: Status {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except FileNotFoundError:
        print(f"❌ File not found: {pdf_path}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_query(question, top_k=3):
    print(f"\n💬 Testing Query...")
    print(f"   Question: '{question}'")
    
    payload = {"question": question, "top_k": top_k}
    
    print("   ⏳ Generating answer...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query successful! (took {elapsed:.2f}s)")
            print(f"\n📝 Answer:")
            print(f"   {data['answer']}")
            print(f"\n📚 Citations:")
            for citation in data['citations']:
                print(f"   {citation}")
            return True
        elif response.status_code == 404:
            print(f"❌ No relevant documents found")
            return False
        else:
            print(f"❌ Query failed: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def run_full_test_suite():
    print_header("🚀 TEST SUITE")
    
    if not test_health_check():
        print("\n⚠️  Server not running.")
        print("   Start with: python main.py")
        return
    
    time.sleep(1)
    
    print_header("📄 DOCUMENT INGESTION")
    pdf_path = input("\nEnter PDF path (or press Enter to skip): ").strip()
    
    if pdf_path:
        if test_document_ingestion(pdf_path):
            time.sleep(2)
            print_header("💬 QUERY TESTING")
            test_query("What is the main topic of this document?")
    else:
        print("\n⚠️  Skipping ingestion test.")
        test_query("What information is available?")
    
    print_header("✨ COMPLETED")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "health":
            test_health_check()
        elif command == "ingest" and len(sys.argv) > 2:
            test_health_check()
            test_document_ingestion(sys.argv[2])
        elif command == "query" and len(sys.argv) > 2:
            test_health_check()
            test_query(" ".join(sys.argv[2:]))
        else:
            print("Usage:")
            print("  python test_api.py")
            print("  python test_api.py health")
            print("  python test_api.py ingest <pdf>")
            print("  python test_api.py query <question>")
    else:
        run_full_test_suite()