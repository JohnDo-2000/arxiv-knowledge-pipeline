import sys, json
from pathlib import Path
sys.path.insert(0, 'src')
from etl.pipeline import process_document
from vectorstore.qdrant_client import VectorStore

store = VectorStore(url='http://localhost:6333')
store.ensure_collection(recreate=True)

raw_dir = Path('data/raw')
total = 0
for pdf in sorted(raw_dir.glob('*.pdf')):
    meta_path = raw_dir / f'{pdf.stem}.metadata.json'
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    else:
        metadata = {'arxiv_id': pdf.stem, 'title': pdf.stem}
    n = process_document(pdf, document_metadata=metadata, vector_store=store)
    print(f"{pdf.stem}: {n} chunks | {metadata.get('title', '?')[:60]}")
    total += n

print(f'\nTotal chunks: {total}')
print(f'Collection count: {store.count()}')
