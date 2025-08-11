#!/usr/bin/env python3
"""
Agentic RAG System with CLI Interface
Supports PDF and JSON documents using Google's generative AI for embeddings.
Features a configurable main LLM (Ollama or OpenRouter), a dedicated OpenRouter LLM for JSON processing,
and specialized agents for PDF parsing (via RAG), JSON parsing, and web search (via Azure Search).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging
import requests # For Azure Search API calls

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.chat_models import ChatOllama # For local Ollama LLM
from langchain_openai import ChatOpenAI # For OpenRouter (OpenAI-compatible) LLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
# from langchain import hub # Removed to use hardcoded prompt for consistency

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, docs_folder: str = "documents"):
        self.docs_folder = Path(docs_folder)
        self.docs_folder.mkdir(exist_ok=True)
        
    def load_documents(self) -> List[Document]:
        """Load all PDF and JSON documents from the documents folder"""
        documents = []
        
        # Process PDF files
        pdf_files = list(self.docs_folder.glob("*.pdf"))
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                # Add metadata
                for doc in docs:
                    doc.metadata.update({
                        "source_file": pdf_file.name,
                        "file_type": "pdf"
                    })
                documents.extend(docs)
                logger.info(f"Loaded PDF: {pdf_file.name} ({len(docs)} pages)")
            except Exception as e:
                logger.error(f"Error loading PDF {pdf_file}: {e}")
        
        # Process JSON files
        json_files = list(self.docs_folder.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert JSON to text representation for initial indexing in vector store
                # The JSONQueryTool will load and parse the raw JSON file when invoked.
                json_text = self._json_to_text(data, json_file.name)
                doc = Document(
                    page_content=json_text,
                    metadata={
                        "source_file": json_file.name,
                        "file_type": "json"
                    }
                )
                documents.append(doc)
                logger.info(f"Loaded JSON: {json_file.name}")
            except Exception as e:
                logger.error(f"Error loading JSON {json_file}: {e}")
        
        return documents
    
    def _json_to_text(self, data: Any, filename: str) -> str:
        """Convert JSON data to a flattened searchable text representation"""
        text_parts = []
        def flatten_json(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_prefix = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, (dict, list)):
                        flatten_json(v, new_prefix)
                    else:
                        text_parts.append(f"{new_prefix}: {v}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_prefix = f"{prefix}[{i}]" if prefix else f"item_{i}"
                    if isinstance(item, (dict, list)):
                        flatten_json(item, new_prefix)
                    else:
                        text_parts.append(f"{new_prefix}: {item}")
            else:
                text_parts.append(f"{prefix}: {obj}")

        flatten_json(data)
        return f"File: {filename}\n\n" + "\n".join(text_parts)

class RAGSystem:
    """Main RAG system with agentic capabilities and multiple LLM integrations"""
    
    def __init__(self, google_api_key: str, docs_folder: str = "documents",
                 llm_provider: str = "ollama", 
                 ollama_model_name: str = "gpt-oss:20b",
                 ollama_base_url: str = "http://localhost:11434",
                 openrouter_api_key: str = None,
                 openrouter_json_llm_model_name: str = "mistralai/mistral-7b-instruct", # Dedicated LLM for JSON summarization
                 azure_search_api_key: str = None,
                 azure_search_endpoint: str = None): # Azure Search endpoint
        
        self.google_api_key = google_api_key
        self.docs_folder = Path(docs_folder)
        self.llm_provider = llm_provider
        self.ollama_model_name = ollama_model_name
        self.ollama_base_url = ollama_base_url
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_json_llm_model_name = openrouter_json_llm_model_name
        self.azure_search_api_key = azure_search_api_key
        self.azure_search_endpoint = azure_search_endpoint

        # Initialize Main LLM component based on provider
        if self.llm_provider == "openrouter":
            if not self.openrouter_api_key:
                raise ValueError("OpenRouter API key must be provided when llm-provider is 'openrouter'.")
            self.main_llm = ChatOpenAI(
                model="openai/gpt-oss-20b", # Specific OpenRouter model name for GPT-OSS
                openai_api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1", # OpenRouter API endpoint
                temperature=0.2 # Slightly increased temperature for better adherence
            )
            logger.info(f"Initialized Main LLM with OpenRouter: openai/gpt-oss-20b")
        else: # Default to Ollama
            self.main_llm = ChatOllama(
                model=self.ollama_model_name,
                base_url=self.ollama_base_url,
                temperature=0.1
            )
            logger.info(f"Initialized Main LLM with Ollama: {self.ollama_model_name} at {self.ollama_base_url}")
        
        # Initialize Dedicated LLM for JSON processing (always OpenRouter for this config)
        if not self.openrouter_api_key:
             raise ValueError("OpenRouter API key is required for the JSON processing LLM.")
        self.json_processing_llm = ChatOpenAI(
            model=self.openrouter_json_llm_model_name,
            openai_api_key=self.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1 # Keep this lower for factual summarization
        )
        logger.info(f"Initialized JSON Processing LLM with OpenRouter: {self.openrouter_json_llm_model_name}")

        # Initialize Google Generative AI Embeddings component (always Google)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.google_api_key
        )
        logger.info(f"Initialized Google Embeddings: models/text-embedding-004")
        
        # Initialize document processor
        self.doc_processor = DocumentProcessor(docs_folder)
        
        # Initialize components
        self.vectorstore = None
        self.qa_chain = None
        self.agent = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Setup the system
        self._setup_system()
    
    def _setup_system(self):
        """Setup the RAG system with vector store and agent"""
        logger.info("Setting up RAG system...")
        
        # Load and process documents
        documents = self.doc_processor.load_documents()
        
        if not documents:
            logger.warning("No documents found! Please add PDF or JSON files to the documents folder.")
        
        if documents:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} document chunks")
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            logger.info("Vector store created")
            
            # Create QA chain
            qa_prompt = PromptTemplate(
                template="""Use the following pieces of context to answer the question at the end. 
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                
                Context: {context}
                
                Question: {question}
                
                Answer: """,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.main_llm, # Use main LLM for general QA
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": qa_prompt}
            )
        else:
            logger.warning("No documents for vector store. Document QA and Search tools will be limited.")
            self.vectorstore = None
            self.qa_chain = None # No QA chain if no documents

        # Create tools for the agent
        tools = [
            Tool(
                name="Document_QA",
                func=self._answer_question,
                description="Use this tool to answer general questions based on the loaded PDF documents or general information from JSON. Input should be a question."
            ),
            Tool(
                name="JSON_Query",
                func=self._query_json_document,
                description="Use this tool to extract specific, structured information about incidents, devices, or users from local JSON documents (e.g., mde.json). "
                            "Input MUST be a JSON string with 'file_name' and 'query' keys. "
                            "The 'query' can be an incident ID (e.g., 66594), a device ID, a user email, or a clear description of what details to extract. "
                            "Example Input: {\"file_name\": \"mde.json\", \"query\": \"summary of incident 66594 including email details\"}"
            )
        ]

        # Add Web Search tool if Azure Search is configured
        if self.azure_search_api_key and self.azure_search_endpoint:
            tools.append(
                Tool(
                    name="Web_Search",
                    func=self._perform_azure_search,
                    description="Use this tool to perform a web search for external or real-time information. "
                                "Input should be a concise search query."
                )
            )
            logger.info("Azure Web Search tool enabled.")
        else:
            logger.warning("Azure Search API key or endpoint not provided. Web Search tool will not be available.")

        # Hardcoded, explicit ReAct prompt for better adherence
        agent_prompt_template = PromptTemplate.from_template("""
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: You should always think about what to do. Prioritize local documents first (Document_QA, JSON_Query). Use Web_Search only for information not found locally or for real-time data. When querying JSON, ensure the input to JSON_Query is always a valid JSON string with 'file_name' and 'query'.
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action (must be formatted correctly for the selected tool, especially for JSON input)
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Question: {input}
        Thought: {agent_scratchpad}
        """)
        
        # Create agent and executor
        self.agent = AgentExecutor(
            agent=create_react_agent(self.main_llm, tools, agent_prompt_template), # Use main_llm for agent reasoning
            tools=tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
        
        logger.info("RAG system setup complete!")
    
    def _answer_question(self, question: str) -> str:
        """Answer a question using the QA chain (primarily for PDF documents)"""
        try:
            if not self.qa_chain:
                return "No documents available for QA. Please add documents to the documents folder."
            
            result = self.qa_chain.invoke({"query": question})
            return result["result"]
        except Exception as e:
            logger.error(f"Error answering question with Document QA: {e}")
            return f"Error with Document QA: {e}"
    
    def _search_documents(self, query: str) -> str:
        """Search documents for relevant information (across all indexed documents)"""
        try:
            if not self.vectorstore:
                return "No documents loaded for search. Please add documents to the documents folder."
            
            docs = self.vectorstore.similarity_search(query, k=3)
            if not docs:
                return "No relevant information found in local documents."
            
            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source_file", "Unknown")
                content_snippet = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                results.append(f"{i}. Source: {source}\nContent: {content_snippet}\n")
            
            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return f"Error searching documents: {e}"

    def _query_json_document(self, input_json_str: str) -> str:
        """
        Extracts specific information from a JSON document and summarizes it using a dedicated LLM.
        Input should be a JSON string with 'file_name' and 'query'.
        The 'query' can be an incident ID, a key, a path, or a description of what to extract.
        """
        try:
            input_dict = json.loads(input_json_str)
            file_name = input_dict.get("file_name")
            query = input_dict.get("query")

            if not file_name or not query:
                return "Error (JSON Query): Both 'file_name' and 'query' must be provided in the JSON input."

            json_file_path = self.docs_folder / file_name
            if not json_file_path.exists():
                return f"Error (JSON Query): JSON file '{file_name}' not found in '{self.docs_folder}'."

            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Find the raw data using the helper function
            raw_result = self._find_in_json(data, query)
            
            if raw_result:
                # Use the dedicated JSON processing LLM to summarize the raw result
                prompt_template = f"Given the following raw JSON data related to a query '{query}', provide a concise and human-readable summary:\n\nJSON Data: {json.dumps(raw_result, indent=2)}\n\nSummary:"
                response = self.json_processing_llm.invoke(prompt_template)
                return f"Summary from {file_name} for query '{query}': {response.content}"
            else:
                return f"No specific information found for '{query}' in '{file_name}'."

        except json.JSONDecodeError:
            return "Error (JSON Query): Invalid JSON input for JSON Query tool. Please provide a valid JSON string. Ensure it's a single line JSON string."
        except Exception as e:
            logger.error(f"Error querying JSON document: {e}", exc_info=True)
            return f"Error querying JSON document: {e}"

    def _find_in_json(self, data: Any, query: str) -> Any:
        """
        Recursive helper to find data in JSON. Improved to handle incident IDs and specific keywords.
        """
        incident_id = None
        if isinstance(query, str) and "incident" in query.lower():
            try:
                incident_id = int(''.join(filter(str.isdigit, query)))
            except ValueError:
                pass

        if isinstance(data, dict):
            # Prioritize finding by incidentId if it's explicitly sought and matches
            if incident_id and (data.get("incidentId") == incident_id or 
                                (isinstance(data.get("properties"), dict) and data["properties"].get("incidentId") == incident_id) or
                                (isinstance(data.get("alerts"), list) and any(alert.get("incidentId") == incident_id for alert in data["alerts"]))):
                
                # If mde.json is structured with 'value' being a list of incidents
                if "value" in data and isinstance(data["value"], list):
                    for item in data["value"]:
                        if isinstance(item, dict) and "alerts" in item and isinstance(item["alerts"], list):
                            for alert in item["alerts"]:
                                if alert.get("incidentId") == incident_id:
                                    return item # Return the whole top-level incident object
                return data

            # Direct key match
            if query in data:
                return data[query]
            
            # Recursive search for relevant values or sub-dictionaries/lists
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    found = self._find_in_json(v, query)
                    if found:
                        return found
                elif isinstance(v, str) and query.lower() in v.lower():
                    return {k:v} # Return key-value pair if value contains query
                elif isinstance(query, str) and query.lower() == str(v).lower():
                     return {k:v} # Exact match for value

        elif isinstance(data, list):
            for item in data:
                # If an incident ID was extracted and found within an item's sub-structure
                if incident_id and isinstance(item, dict) and (item.get("incidentId") == incident_id or 
                                                              (isinstance(item.get("properties"), dict) and item["properties"].get("incidentId") == incident_id) or
                                                              (isinstance(item.get("alerts"), list) and any(alert.get("incidentId") == incident_id for alert in item["alerts"]))):
                    return item
                
                found = self._find_in_json(item, query)
                if found:
                    return found
        
        elif isinstance(data, str):
            if isinstance(query, str) and query.lower() in data.lower():
                return data

        return None # No match found


    def _perform_azure_search(self, query: str) -> str:
        """
        Performs a web search using Azure AI Search.
        Input is a search query string.
        """
        if not self.azure_search_api_key or not self.azure_search_endpoint:
            return "Error: Azure Search API key or endpoint not configured."

        # Note: This URL assumes you are searching directly against the service,
        # which might not be what you want. Typically, you search against a specific index:
        # f"{self.azure_search_endpoint}/indexes/<your-index-name>/docs/search?api-version=2023-11-01"
        # For this to work robustly, you'd likely need to know the index name or
        # adjust the endpoint to point to a specific index's search path.
        # For a general search, using the /indexes endpoint as a discovery might work
        # but a /docs/search endpoint on a specific index is more common.
        # Assuming you have an index where results are stored, you'd want something like:
        # search_url = f"{self.azure_search_endpoint}/indexes/my-search-index/docs/search?api-version=2023-11-01"
        # For now, I'll keep the general /indexes endpoint as provided, but note its limitations.

        # Corrected URL for searching a *specific index*. You must replace "your-index-name"
        # with the actual name of your Azure Search index.
        # This is a crucial step for the Azure Search to work correctly.
        # If you don't have a specific index, you'll need to create one and index your data into it.
        # For a basic search over existing data, replace "your-index-name" with your actual index name.
        # Example: search_url = f"{self.azure_search_endpoint}/indexes/my-incident-index/docs/search?api-version=2023-11-01"
        # Since I don't know your index name, I'll use a placeholder and emphasize this in the conclusion.
        # For general web search *outside* your indexed data, consider Azure AI Search's capability
        # to index public web content or integrate with Bing Search API.
        # The current implementation queries *your own data* indexed in Azure AI Search.

        # IMPORTANT: Replace 'YOUR_AZURE_SEARCH_INDEX_NAME' with your actual index name in Azure AI Search.
        # This is CRITICAL for the Azure Search tool to function.
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "your-default-azure-search-index-name") # Add env var for index name
        search_url = f"{self.azure_search_endpoint}/indexes/{index_name}/docs/search?api-version=2023-11-01"
        
        headers = {
            "api-key": self.azure_search_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "search": query,
            "queryType": "simple", # Or "full", "semantic", etc. depending on your index config
            "top": 3 # Number of results to retrieve
        }

        try:
            response = requests.post(search_url, headers=headers, json=payload, timeout=15) # Increased timeout
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            search_results = response.json()

            if not search_results or not search_results.get("value"):
                return "No relevant web search results found."
            
            formatted_results = []
            for i, result in enumerate(search_results["value"], 1):
                # Customize this based on the actual structure of your Azure Search results
                # These are common fields, adjust as per your index schema
                title = result.get("title", result.get("name", "No Title"))
                content = result.get("content", result.get("description", "No Content"))
                url = result.get("url", "No URL")
                
                formatted_results.append(f"{i}. Title: {title}\nURL: {url}\nSnippet: {content[:300]}...\n") # Truncate snippet
            
            return "\n".join(formatted_results)

        except requests.exceptions.HTTPError as http_err:
            return f"HTTP error during Azure Search: {http_err} - Response: {response.text}"
        except requests.exceptions.ConnectionError as conn_err:
            return f"Connection error during Azure Search: {conn_err}. Check endpoint and network."
        except requests.exceptions.Timeout as timeout_err:
            return f"Timeout error during Azure Search: {timeout_err}. Request took too long."
        except json.JSONDecodeError:
            return "Error: Failed to decode JSON response from Azure Search."
        except Exception as e:
            logger.error(f"An unexpected error occurred during Azure Search: {e}", exc_info=True)
            return f"An unexpected error occurred during Azure Search: {e}"
    
    def chat(self, message: str) -> str:
        """Process a chat message through the agent"""
        try:
            if not self.agent:
                return "System not initialized. Please check if documents are loaded."
            
            response = self.agent.invoke({"input": message})
            return response["output"]
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return f"Error processing message: {e}"
    
    def refresh_documents(self):
        """Refresh the document store with new documents"""
        logger.info("Refreshing document store...")
        self._setup_system()

class RAGChatbot:
    """CLI chatbot interface"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.running = True
    
    def run(self):
        """Run the interactive chatbot"""
        print("\n" + "="*60)
        print("🤖 Advanced Agentic RAG System CLI Chatbot")
        print("="*60)
        
        if self.rag_system.llm_provider == "openrouter":
            print(f"Main LLM: OpenRouter (Model: openai/gpt-oss-20b)")
        else:
            print(f"Main LLM: Ollama (Model: {self.rag_system.ollama_model_name} at {self.rag_system.ollama_base_url})")
        print(f"JSON Processing LLM: OpenRouter (Model: {self.rag_system.openrouter_json_llm_model_name})")
        print(f"Embeddings: Google (Model: models/text-embedding-004)")
        if self.rag_system.azure_search_api_key:
            print(f"Web Search: Azure Search enabled (Endpoint: {self.rag_system.azure_search_endpoint})")
        else:
            print(f"Web Search: Azure Search disabled (missing key/endpoint)")
        print("="*60)
        print("Type 'help' for commands, 'quit' to exit")
        print("="*60 + "\n")
        
        while self.running:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    self.running = False
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'refresh':
                    print("\n🔄 Refreshing documents...")
                    self.rag_system.refresh_documents()
                    print("✅ Documents refreshed!")
                    continue
                
                elif user_input.lower() == 'status':
                    self._show_status()
                    continue
                
                # Process the query
                print("\n🤔 Thinking...")
                response = self.rag_system.chat(user_input)
                print(f"\n🤖 Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                self.running = False
            except Exception as e:
                print(f"\n❌ Error: An unexpected error occurred: {e}")
                logger.error(f"Error during chat session: {e}", exc_info=True)
    
    def _show_help(self):
        """Show help information"""
        help_text = """
Available commands:
- help: Show this help message
- refresh: Reload documents from the documents folder and rebuild vector store.
- status: Show system status and configurations.
- quit/exit/bye: Exit the chatbot.

You can ask questions about:
- Content in your PDF documents (e.g., "Summarize the key takeaways from the social engineering IRP.").
- Specific incidents or data points in your JSON files (e.g., "What is the detailed summary of incident 66594?"). The agent will use the JSON_Query tool for this.
- General knowledge or real-time information that might require a web search (e.g., "What is the current cybersecurity threat landscape?"). The agent will use the Web_Search tool for this.
        """
        print(help_text)
    
    def _show_status(self):
        """Show system status"""
        docs_folder = Path(self.rag_system.docs_folder)
        pdf_count = len(list(docs_folder.glob("*.pdf")))
        json_count = len(list(docs_folder.glob("*.json")))
        
        main_llm_status_line = ""
        if self.rag_system.llm_provider == "openrouter":
            main_llm_status_line = f"Main LLM: OpenRouter (Model: openai/gpt-oss-20b)"
        else:
            main_llm_status_line = f"Main LLM: Ollama (Model: {self.rag_system.ollama_model_name} at {self.rag_system.ollama_base_url})"

        json_llm_status_line = f"JSON Processing LLM: OpenRouter (Model: {self.rag_system.openrouter_json_llm_model_name})"
        
        azure_search_status = "Enabled" if self.rag_system.azure_search_api_key and self.rag_system.azure_search_endpoint else "Disabled"
        azure_search_endpoint_display = self.rag_system.azure_search_endpoint if self.rag_system.azure_search_endpoint else "N/A"

        status = f"""
📊 System Status:
- Documents folder: {docs_folder.absolute()}
- PDF files: {pdf_count}
- JSON files: {json_count}
- {main_llm_status_line}
- {json_llm_status_line}
- Google Embeddings: models/text-embedding-004
- Vector store: {'✅ Ready' if self.rag_system.vectorstore else '❌ Not loaded'}
- Agent: {'✅ Ready' if self.rag_system.agent else '❌ Not initialized'}
- Web Search (Azure): {azure_search_status} (Endpoint: {azure_search_endpoint_display})
        """
        print(status)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Advanced Agentic RAG System CLI")
    parser.add_argument("--docs-folder", default="documents", help="Path to documents folder (default: 'documents')")
    parser.add_argument("--google-api-key", help="Google API key for embeddings (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--llm-provider", 
                        default="ollama", 
                        choices=["ollama", "openrouter"],
                        help="Choose the main LLM provider: 'ollama' (local) or 'openrouter' (default: 'ollama').")
    parser.add_argument("--ollama-model", 
                        default="gpt-oss:20b", 
                        help="Name of the Ollama model to use when --llm-provider is 'ollama' (default: 'gpt-oss:20b').")
    parser.add_argument("--ollama-base-url", 
                        default="http://localhost:11434",
                        help="Base URL for the Ollama API when --llm-provider is 'ollama' (default: 'http://localhost:11434').")
    parser.add_argument("--openrouter-api-key", 
                        help="OpenRouter API key for both OpenRouter LLMs (or set OPENROUTER_API_KEY env var). Required if --llm-provider is 'openrouter' or if OpenRouter JSON LLM is used.")
    parser.add_argument("--openrouter-json-llm-model-name",
                        default="mistralai/mistral-7b-instruct", # Default for JSON processing LLM
                        help="OpenRouter model to use specifically for JSON parsing/summarization (default: 'mistralai/mistral-7b-instruct').")
    parser.add_argument("--azure-search-api-key", help="Azure AI Search API key (or set AZURE_SEARCH_API_KEY env var)")
    parser.add_argument("--azure-search-endpoint", help="Azure AI Search service endpoint (e.g., https://<your-service-name>.search.windows.net, or set AZURE_SEARCH_ENDPOINT env var)")
    parser.add_argument("--azure-search-index-name", # New argument for Azure Search index name
                        help="Name of the Azure AI Search index to query (or set AZURE_SEARCH_INDEX_NAME env var). CRITICAL for Web Search tool.")
    
    args = parser.parse_args()
    
    # Get Google API key for embeddings
    google_api_key = args.google_api_key or os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("❌ Error: Google API key not provided for embeddings!")
        print("Please provide a Google API key using --google-api-key argument or set GOOGLE_API_KEY environment variable.")
        sys.exit(1)
    
    # Get OpenRouter API key (used for both OpenRouter LLMs if chosen)
    openrouter_api_key = args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    # Validate OpenRouter key if OpenRouter is chosen as primary LLM or if JSON LLM is using OpenRouter
    if (args.llm_provider == "openrouter" or args.openrouter_json_llm_model_name) and not openrouter_api_key:
        print("❌ Error: OpenRouter API key not provided!")
        print("Please provide an OpenRouter API key using --openrouter-api-key argument or set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    # Get Azure Search API key, endpoint, and index name
    azure_search_api_key = args.azure_search_api_key or os.getenv("AZURE_SEARCH_API_KEY")
    azure_search_endpoint = args.azure_search_endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
    azure_search_index_name = args.azure_search_index_name or os.getenv("AZURE_SEARCH_INDEX_NAME")

    # Note: We don't exit if Azure Search keys/endpoint/index are missing, as the system can still run without web search.
    if not azure_search_api_key or not azure_search_endpoint or not azure_search_index_name:
        logger.warning("Azure Search API key, endpoint, or index name missing. Web Search tool will be disabled.")
        # Ensure they are None if incomplete, so the tool itself returns an error
        azure_search_api_key = None
        azure_search_endpoint = None
        azure_search_index_name = None # Important: Reset if not fully provided
    
    try:
        # Initialize RAG system
        print(f"🚀 Initializing Advanced Agentic RAG system...")
        rag_system = RAGSystem(
            google_api_key=google_api_key,
            docs_folder=args.docs_folder,
            llm_provider=args.llm_provider,
            ollama_model_name=args.ollama_model,
            ollama_base_url=args.ollama_base_url,
            openrouter_api_key=openrouter_api_key,
            openrouter_json_llm_model_name=args.openrouter_json_llm_model_name,
            azure_search_api_key=azure_search_api_key,
            azure_search_endpoint=azure_search_endpoint
            # Note: azure_search_index_name is handled directly within _perform_azure_search method
        )
        # Pass the index name to the RAGSystem instance for the Azure Search method
        # We need a way to pass this specific argument if it's not part of the __init__
        # For simplicity, let's store it directly on the RAGSystem instance if provided
        rag_system.azure_search_index_name = azure_search_index_name # Attach index name to instance

        # Start chatbot
        chatbot = RAGChatbot(rag_system)
        chatbot.run()
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
