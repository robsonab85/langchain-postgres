import os


def get_embeddings():
    """Retorna o provider de embeddings configurado via variáveis de ambiente.

    Prioriza OpenAI quando OPENAI_API_KEY estiver definida; caso contrário,
    usa Google Gemini quando GOOGLE_API_KEY estiver definida.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openai_key:
        from langchain_openai import OpenAIEmbeddings

        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model, api_key=openai_key)

    if google_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=google_key)

    raise RuntimeError(
        "Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env para configurar os embeddings."
    )


def get_llm():
    """Retorna o LLM configurado via variáveis de ambiente.

    Prioriza OpenAI quando OPENAI_API_KEY estiver definida; caso contrário,
    usa Google Gemini quando GOOGLE_API_KEY estiver definida.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openai_key:
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
        return ChatOpenAI(model=model, api_key=openai_key)

    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite")
        return ChatGoogleGenerativeAI(model=model, google_api_key=google_key)

    raise RuntimeError(
        "Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env para configurar o LLM."
    )
