# Ingestão e Busca Semântica com LangChain e Postgres

Desafio MBA Engenharia de Software com IA - Full Cycle.

Sistema de CLI que ingere um PDF em um banco vetorial (PostgreSQL + pgVector) e responde perguntas do usuário com base **apenas** no conteúdo desse documento, usando LangChain e um LLM (OpenAI ou Google Gemini).

## Arquitetura

- **Ingestão** (`src/ingest.py`): lê o PDF, divide em chunks de 1000 caracteres com overlap de 150, gera embeddings e grava no PostgreSQL via pgVector.
- **Busca** (`src/search.py`): vetoriza a pergunta, busca os 10 trechos mais relevantes (`k=10`) no banco vetorial, monta o prompt e consulta o LLM.
- **Chat** (`src/chat.py`): CLI que faz o loop de perguntas e respostas no terminal.
- **Providers** (`src/providers.py`): seleciona automaticamente o provedor (OpenAI ou Google Gemini) com base nas chaves definidas no `.env`.

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose
- Uma chave de API da [OpenAI](https://platform.openai.com/api-keys) **ou** do [Google AI Studio](https://aistudio.google.com/app/apikey)

## Configuração

1. Crie e ative um ambiente virtual:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Copie o arquivo de variáveis de ambiente e preencha com sua chave de API (apenas um dos provedores é necessário):

   ```bash
   cp .env.example .env
   ```

   | Variável | Descrição |
   |---|---|
   | `OPENAI_API_KEY` | Chave da API OpenAI (usa `text-embedding-3-small` e `gpt-5-nano`) |
   | `GOOGLE_API_KEY` | Chave da API Google (usa `models/gemini-embedding-001` e `gemini-3.5-flash-lite`) |
   | `DATABASE_URL` | String de conexão do PostgreSQL |
   | `PG_VECTOR_COLLECTION_NAME` | Nome da coleção usada no pgVector |
   | `PDF_PATH` | Caminho do PDF a ser ingerido |

## Execução

1. Subir o banco de dados:

   ```bash
   docker compose up -d
   ```

2. Executar a ingestão do PDF:

   ```bash
   python src/ingest.py
   ```

3. Rodar o chat:

   ```bash
   python src/chat.py
   ```

## Exemplo de uso

```
Faça sua pergunta (digite 'sair' para encerrar):

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: sair
```

## Solução de problemas

- **`404 ... is not found for API version v1beta`**: o modelo de embedding/LLM do Gemini configurado foi descontinuado para sua conta. Liste os modelos disponíveis para sua chave e atualize `GOOGLE_EMBEDDING_MODEL`/`GOOGLE_MODEL` no `.env`:

  ```bash
  curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=SUA_CHAVE" | grep '"name"'
  ```

- **`429 ResourceExhausted` / quota excedida**: a camada gratuita do Gemini tem limite de requisições por minuto. O `src/ingest.py` já envia os chunks em lotes pequenos com retry automático; se mesmo assim persistir, aumente `RETRY_DELAY_SECONDS` ou diminua `BATCH_SIZE` no início do arquivo.

## Estrutura do projeto

```
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py
│   ├── search.py
│   ├── chat.py
│   └── providers.py
├── document.pdf
└── README.md
```
