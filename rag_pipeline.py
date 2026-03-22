import ollama
import chromadb
import fitz
import os

# ---- DYNAMIC PATHS ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.join(BASE_DIR, "docs")
DB_FOLDER = os.path.join(BASE_DIR, "db")

# ---- SETUP ----
def setup():
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    os.makedirs(DB_FOLDER, exist_ok=True)
    print(f"Drop your PDF files into: {DOCS_FOLDER}")

# ---- STEP 1: Extract text from PDFs ----
def load_pdfs(folder_path):
    documents = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        print("Add your PDF files to the docs/ folder and run again.")
        return []

    for filename in pdf_files:
        filepath = os.path.join(folder_path, filename)
        try:
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text()
            documents.append({
                'filename': filename,
                'text': text
            })
            print(f"Loaded: {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return documents

# ---- STEP 2: Store in vector database ----
def build_knowledge_base(collection, documents):
    for i, doc in enumerate(documents):
        text = doc['text']
        chunks = [text[j:j+500] for j in range(0, len(text), 500)]
        collection.add(
            documents=chunks,
            ids=[f"{i}_{k}" for k in range(len(chunks))]
        )
    print(f"Knowledge base built — {collection.count()} chunks indexed")

# ---- STEP 3: Ask questions ----
def ask(collection, question):
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    if not results['documents'][0]:
        return "No relevant information found in your documents."

    context = "\n".join(results['documents'][0])

    response = ollama.chat(
        model='llama3.1',
        messages=[{
            'role': 'user',
            'content': f"""Use the following notes to answer the question.
Only answer based on the provided notes. If the answer isn't in the notes,
say so clearly.

Notes:
{context}

Question: {question}"""
        }]
    )

    return response['message']['content']

# ---- MAIN ----
def main():
    setup()

    # Connect to persistent vector database
    client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = client.get_or_create_collection("documents")

    # Only index PDFs if database is empty
    if collection.count() == 0:
        print("Building knowledge base for the first time...")
        documents = load_pdfs(DOCS_FOLDER)
        if not documents:
            return
        build_knowledge_base(collection, documents)
        print("Done — knowledge base saved. Won't need to rebuild unless you add new PDFs.")
    else:
        print(f"Knowledge base loaded — {collection.count()} chunks ready")
        print(f"Add new PDFs to {DOCS_FOLDER} and delete the db/ folder to rebuild")

    print("\nReady! Ask questions about your documents.")
    print("Type 'quit' to exit\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        answer = ask(collection, question)
        print(f"\nLLaMA: {answer}\n")

if __name__ == "__main__":
    main()