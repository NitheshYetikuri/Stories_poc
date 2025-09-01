import os
from dotenv import load_dotenv
import json
import shutil
from pathlib import Path
import git
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from javalang.parse import parse
from javalang.tree import (
    ClassDeclaration,
    MethodDeclaration,
    FieldDeclaration,
    MemberReference,
    MethodInvocation,
    VariableDeclarator,
    InterfaceDeclaration,
    Type,
    ClassCreator,
)

# -------------------- CONFIG --------------------
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PROJECTS_DIR = r"C:\Users\yetik\GenAI\DevStoryAI-main\DevStoryAI\projects"
OUTPUT_PATH = "metadata/classesmetadata.json"
# ------------------------------------------------


# --------------- YOUR PARSING LOGIC (unchanged) ---------------
STANDARD_LIBRARIES = {'System', 'log', 'e'}

def parse_java_file(file_path):
    """Parses a Java file and returns the abstract syntax tree."""
    with open(file_path, 'r') as file:
        return parse(file.read())

def is_standard_library_call(method_call):
    """Checks if a method call belongs to a standard library."""
    return any(method_call.startswith(lib) for lib in STANDARD_LIBRARIES)

def extract_method_calls(method_node):
    """Extracts method calls from a method's AST node."""
    calls = []
    for _, child in method_node:
        if isinstance(child, MethodInvocation):
            method_call = child.member
            if child.qualifier:
                qualifier = child.qualifier
                if isinstance(qualifier, MemberReference):
                    qualifier = qualifier.member
                elif isinstance(qualifier, ClassCreator) and isinstance(qualifier.type, Type):
                    qualifier = qualifier.type.name
                method_call = f"{qualifier}.{method_call}"
            if not is_standard_library_call(method_call):
                calls.append(method_call)
    return calls

def get_type_name(type_node):
    """Extracts the name from a type node."""
    return type_node.name if isinstance(type_node, Type) else str(type_node) if type_node else None

def extract_relationships(tree, file_path):
    """Extracts class relationships and metadata from a Java AST."""
    relationships = {}
    for _, node in tree:
        if isinstance(node, (ClassDeclaration, InterfaceDeclaration)):
            node_type = 'class' if isinstance(node, ClassDeclaration) else 'interface'
            name = node.name
            extends = [get_type_name(ext) for ext in node.extends] if hasattr(node, 'extends') and node.extends else None
            implements = [get_type_name(impl) for impl in node.implements] if hasattr(node, 'implements') and node.implements else []
            attributes = []
            methods = {}

            for member in node.body:
                if isinstance(member, FieldDeclaration):
                    try:
                        for declarator in member.declarators:
                            if isinstance(declarator, VariableDeclarator):
                                attributes.append({
                                    'name': declarator.name,
                                    'type': get_type_name(member.type),
                                    'modifiers': list(member.modifiers)
                                })
                    except AttributeError as e:
                        print(f"Error extracting field in {name}: {e}")
                        continue
                elif isinstance(member, MethodDeclaration):
                    try:
                        parameters = [{'name': p.name, 'type': get_type_name(p.type), 'modifiers': list(p.modifiers)} for p in member.parameters]
                        return_type = get_type_name(member.return_type)
                        methods[member.name] = {
                            'calls': extract_method_calls(member),
                            'parameters': parameters,
                            'return_type': return_type,
                            'modifiers': list(member.modifiers)
                        }
                    except AttributeError as e:
                        print(f"Error extracting method {member.name} in {name}: {e}")
                        continue

            ordered_info = {
                'Class Name': name,
                'type': node_type,
                'Class Path': file_path,
                'extends': extends if extends or (isinstance(extends, list) and extends) else None,
                'implements': implements if implements else None,
                'attributes': attributes if attributes else None,
                'methods': methods if methods else None
            }
            relationships[name] = ordered_info
    return relationships

def find_java_files(base_dir):
    """Finds all Java files in a directory and its subdirectories."""
    java_files = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".java"):
                class_name = file[:-5]
                java_files[class_name] = os.path.join(root, file)
    return java_files

def recursive_parse(class_name, java_files, parsed_classes):
    """Recursively parses a class and its dependencies."""
    if class_name in parsed_classes:
        return parsed_classes[class_name]

    file_path = java_files.get(class_name)
    if not file_path:
        return {}

    try:
        tree = parse_java_file(file_path)
        relationships = extract_relationships(tree, file_path)
        if class_name in relationships:
            parsed_classes[class_name] = relationships[class_name]
        else:
            for name, data in relationships.items():
                parsed_classes[name] = data
                class_name = name
                break
            else:
                return {}

        if class_name in parsed_classes and parsed_classes[class_name].get('methods'):
            for method_info in parsed_classes[class_name]['methods'].values():
                if method_info.get('calls'):
                    for call in method_info['calls']:
                        if '.' in call:
                            qualifier, _ = call.split('.', 1)
                            recursive_parse(qualifier, java_files, parsed_classes)

        if class_name in parsed_classes:
            node_info = parsed_classes[class_name]
            for parent in node_info.get('extends', []) or []:
                recursive_parse(parent, java_files, parsed_classes)
            for implemented in node_info.get('implements', []) or []:
                recursive_parse(implemented, java_files, parsed_classes)

        return parsed_classes.get(class_name, {})

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {}
# ---------------- END PARSING LOGIC (unchanged) ----------------


def clone_and_process_repo(github_url, google_api_key=None):
    """
    Clone the repo into PROJECTS_DIR, parse Java files with your parser,
    write JSON to OUTPUT_PATH, and store embeddings in ChromaDB.
    """
    os.makedirs(PROJECTS_DIR, exist_ok=True)

    repo_name = github_url.rstrip("/").split("/")[-1].replace(".git", "")
    clone_path = os.path.join(PROJECTS_DIR, repo_name)

    if os.path.exists(clone_path):
        print(f"⚠ Repo already exists at {clone_path}, deleting and recloning...")
        shutil.rmtree(clone_path, onerror=remove_readonly)

    print(f"Cloning {github_url} into {clone_path} ...")
    git.Repo.clone_from(github_url, clone_path)

    java_files = find_java_files(clone_path)
    parsed_classes = {}
    relationships_list = []

    for class_name in list(java_files.keys()):
        rel = recursive_parse(class_name, java_files, parsed_classes)
        if rel:
            relationships_list.append(rel)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as json_file:
        json.dump(relationships_list, json_file, indent=4)

    print(f"✅ Metadata saved to {OUTPUT_PATH}")

    # ---------- ChromaDB logic ----------
    api_key_to_use = google_api_key or GOOGLE_API_KEY
    if api_key_to_use:
        print("Creating vector embeddings and storing in ChromaDB...")
        try:
            try:
                client = chromadb.PersistentClient(path="vectordb/chromadb/")
            except AttributeError:
                client = chromadb.Client(persist_directory="vectordb/chromadb/")

            class GoogleEmbeddingFunction(EmbeddingFunction):
                def __init__(self, api_key):
                    self.api_key = api_key
                    self.model = GoogleGenerativeAIEmbeddings(
                        model="models/text-embedding-004",
                        google_api_key=self.api_key
                    )
                def __call__(self, input):
                    return self.model.embed_documents(input)

            gemini_ef = GoogleEmbeddingFunction(api_key=api_key_to_use)

            collection = client.get_or_create_collection(
                name="relations",
                embedding_function=gemini_ef
            )

            data = json.loads(Path(OUTPUT_PATH).read_text())
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
            chunks = splitter.split_text(str(data))
            # Unique IDs per repo to avoid "existing ID" warnings
            ids = [f"{repo_name}_doc_{i}" for i in range(len(chunks))]

            collection.add(documents=chunks, ids=ids)
            print("✅ Vector embeddings stored in ChromaDB 'relations' collection.")
        except Exception as e:
            print(f"❌ Error creating or storing vector embeddings in ChromaDB: {e}")
    else:
        print("⚠ Google API key not provided. Skipping vector store creation.")

def remove_readonly(func, path, exc_info):
    """
    Error handler for shutil.rmtree. It changes the file's permission
    to allow deletion and retries the operation.
    """
    os.chmod(path, 0o777)
    func(path)

def delete_projects_folder():
    """
    Deletes the specified 'projects' directory and all its contents,
    handling common permission errors.
    """
    projects_folder_path = r"C:\Users\yetik\GenAI\DevStoryAI-main\DevStoryAI\projects"
    
    if os.path.exists(projects_folder_path):
        try:
            # Use shutil.rmtree with the custom error handler
            shutil.rmtree(projects_folder_path, onerror=remove_readonly)
            print(f"Successfully deleted the entire '{projects_folder_path}' folder.")
        except OSError as e:
            # Catch other potential errors
            print(f"Error deleting folder: {e.filename} - {e.strerror}.")
    else:
        print(f"The folder '{projects_folder_path}' does not exist.")





if __name__ == "__main__":
    while True:
        print("\n--- MENU ---")
        print("1. Clone GitHub repo, parse, and store in ChromaDB")
        print("2. Delete all cloned repos")
        print("3. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            github_url = input("Enter GitHub repo URL: ").strip()
            clone_and_process_repo(github_url)

        elif choice == "2":
            confirm = input("Are you sure you want to delete all cloned repos? (y/n): ").strip().lower()
           
            delete_projects_folder()

        elif choice == "3":
            break
        else:
            print("Invalid choice, try again.")

