import os
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from providers import get_embeddings

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
DATABASE_URL = os.getenv("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")

# APIs de embeddings gratuitas costumam ter limite de requisições por minuto.
# Enviar os chunks em lotes pequenos, com pausa e retry, evita erros 429.
BATCH_SIZE = 10
RETRY_DELAY_SECONDS = 20
MAX_RETRIES = 5


def ingest_pdf():
    if not PDF_PATH:
        raise RuntimeError("Defina PDF_PATH no arquivo .env.")
    if not DATABASE_URL:
        raise RuntimeError("Defina DATABASE_URL no arquivo .env.")
    if not PG_VECTOR_COLLECTION_NAME:
        raise RuntimeError("Defina PG_VECTOR_COLLECTION_NAME no arquivo .env.")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise RuntimeError(f"Nenhum conteúdo extraído de '{PDF_PATH}'.")

    embeddings = get_embeddings()

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
        pre_delete_collection=True,
    )

    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                vector_store.add_documents(batch)
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                print(
                    f"Erro no lote {start}-{start + len(batch)} "
                    f"(tentativa {attempt}/{MAX_RETRIES}): {e}. "
                    f"Aguardando {RETRY_DELAY_SECONDS}s..."
                )
                time.sleep(RETRY_DELAY_SECONDS)

        print(f"Processado {min(start + BATCH_SIZE, total)}/{total} chunks...")

    print(
        f"Ingestão concluída: {total} chunks armazenados na coleção "
        f"'{PG_VECTOR_COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    ingest_pdf()
