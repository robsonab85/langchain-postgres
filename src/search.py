from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_postgres import PGVector

from providers import get_embeddings, get_llm

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_vector_store():
    if not DATABASE_URL:
        raise RuntimeError("Defina DATABASE_URL no arquivo .env.")
    if not PG_VECTOR_COLLECTION_NAME:
        raise RuntimeError("Defina PG_VECTOR_COLLECTION_NAME no arquivo .env.")

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


def _retrieve_context(pergunta: str) -> dict:
    vector_store = get_vector_store()
    resultados = vector_store.similarity_search_with_score(pergunta, k=10)
    contexto = "\n\n".join(doc.page_content for doc, _score in resultados)
    return {"contexto": contexto, "pergunta": pergunta}


def _build_chain():
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = get_llm()
    return RunnableLambda(_retrieve_context) | prompt | llm | StrOutputParser()


def search_prompt(question: str | None = None):
    """Constrói a chain de busca+resposta.

    Sem argumentos, retorna a chain pronta para uso (chain.invoke(pergunta)).
    Se `question` for informado, já invoca a chain e retorna a resposta.
    """
    try:
        chain = _build_chain()
    except Exception as e:
        print(f"Erro ao inicializar a busca: {e}")
        return None

    if question is not None:
        return chain.invoke(question)

    return chain
