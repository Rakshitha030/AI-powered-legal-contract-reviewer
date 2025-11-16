import os
import re
import sqlite3
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False
import numpy as np
import faiss
from datetime import datetime
from typing import List, Tuple

# External Services
import requests
import json

# Document Processing Dependencies
import pdfplumber
import docx
from PIL import Image
import pytesseract
import pandas as pd

try:
    import easyocr
    _HAS_EASYOCR = True
except Exception:
    _HAS_EASYOCR = False

try:
    from langdetect import detect
    _HAS_LANGDETECT = True
except Exception:
    _HAS_LANGDETECT = False

# Hugging Face Transformers and Sentence Transformers Dependencies
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

# Flask API Dependencies
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# --- CONFIGURATION ---
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
SUMMARIZER_MODEL = "t5-small"
QA_MODEL = "distilbert-base-uncased-distilled-squad"
SAFETY_MODEL = "martin-ha/toxic-comment-model"

# Combined/Refined Constants
SIMILARITY_THRESHOLD_SELECT = 0.52 # Used for Key Sentence selection
MAX_CHUNK_WORDS = 500  # For RAG chunks
T5_MAX_INPUT_LENGTH = 512 # T5-small max input
TOP_K_RAG = 5          # RAG retrieval count
TOP_K_SENT_EXTRACT = 12 # Key sentence extraction count

DB_FILE = "rag_logs.db"
# NOTE: Replace with your actual SerpAPI key or use an environment variable
SERPAPI_KEY = "6388ba07bb8ea7c3dd34e19e4f915c9260784a1a22507cfd12285f89f0f6bbe0"

TRUSTED_DOMAINS = {
    "legal": {"domain": "indiankanoon.org", "keywords": ["act", "law", "court", "section", "case", "judgment", "constitution", "legal", "contract", "agreement", "party"]},
    "medical": {"domain": "who.int", "keywords": ["disease", "treatment", "vaccine", "symptom", "virus", "medical", "health"]},
    "tech": {"domain": "stackoverflow.com", "keywords": ["python", "javascript", "algorithm", "software", "flask", "pytorch", "code", "programming", "api"]},
    "finance": {"domain": "investopedia.com", "keywords": ["stock", "investment", "finance", "bank", "economy", "market", "currency", "trader"]}
}

# Legal Document Linter Rules (From Gradio code)
SUGGESTION_RULES = [
    (r"\b(may|could|might|should)\b", "Consider replacing weak modal verbs with clearer obligations."),
    (r"\b(payment|amount|fee|rupees|\$|\€|\£)\b", "Ensure currency and payment schedule are clear."),
    (r"\b(deadline|due date|within \d+ (days|weeks|months))\b", "Add unambiguous deadlines."),
    (r"\b(terminate|termination)\b", "Check termination clauses."),
    (r"\b(liabilit(y|ies)|indemnif)\b", "Ensure liability/indemnity caps and exclusions are explicit."),
]

# -----------------------------
# RAGChatbot Class: Manages Models, State, and Core Logic
# -----------------------------
class RAGChatbot:
    def __init__(self, serpapi_key, db_file):
        self.SERPAPI_KEY = serpapi_key
        self.DB_FILE = db_file
        # Determine device
        self.device_id = 0 if (_HAS_TORCH and torch.cuda.is_available()) else -1
        self.device_name = 'cuda' if (_HAS_TORCH and torch.cuda.is_available()) else 'cpu'
        self.dtype = (torch.float16 if (_HAS_TORCH and self.device_name == 'cuda') else (torch.float32 if _HAS_TORCH else None))
        
        # Determine device
        self.device_id = 0 if torch.cuda.is_available() else -1
        self.device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16 if self.device_name == 'cuda' else torch.float32

        try:
            print("Loading Sentence Transformer...")
            self.embed_model = SentenceTransformer(EMBEDDING_MODEL, device=self.device_name)
            
            print("Loading T5 for Summarization...")
            self.summ_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL, legacy=False, use_fast=True)
            self.summ_model = AutoModelForSeq2SeqLM.from_pretrained(
                SUMMARIZER_MODEL,
                device_map="auto" if self.device_id >= 0 else None,
                torch_dtype=self.dtype,
            )
            self.summarizer_pipeline = pipeline(
                "summarization", 
                model=self.summ_model, 
                tokenizer=self.summ_tokenizer, 
                framework="pt",
                device=self.device_id,
                max_length=500,
                min_length=30,
                do_sample=False,
            )
            
            print("Loading QA model (DistilBERT)...")
            self.qa_pipeline = pipeline("question-answering", model=QA_MODEL, tokenizer=QA_MODEL, device=self.device_id)
            
            print("Loading Safety model...")
            self.safety_tokenizer = AutoTokenizer.from_pretrained(SAFETY_MODEL)
            self.safety_model = AutoModelForSequenceClassification.from_pretrained(SAFETY_MODEL)
            self.safety_pipeline = pipeline("text-classification", model=self.safety_model, tokenizer=self.safety_tokenizer, device=self.device_id)
            
            self.init_db()
            print(f"RAGChatbot initialized. Models loaded on device: {self.device_name}")

        except Exception as e:
            print("\n!!! CRITICAL MODEL LOADING ERROR !!!")
            print(f"Error details: {e}")
            raise e

    # --- Database (for logging) ---
    def init_db(self):
        conn = sqlite3.connect(self.DB_FILE)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                answer TEXT,
                timestamp TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_query(self, query, answer):
        conn = sqlite3.connect(self.DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO logs (query, answer, timestamp) VALUES (?, ?, ?)",
                  (query, answer, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    # --- Document Processing ---
    def load_document(self, file_storage) -> Tuple[str, str]:
        """Extracts text from various file formats."""
        filename = file_storage.filename
        lower = filename.lower()
        
        temp_dir = "temp_uploads"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        temp_path = os.path.join(temp_dir, filename)
        file_storage.save(temp_path)
        
        text = ""
        file_type = "unknown"
        try:
            if lower.endswith(".pdf"):
                file_type = "pdf"
                with pdfplumber.open(temp_path) as pdf:
                    text = "\n".join([(page.extract_text() or "").strip() for page in pdf.pages]).strip()
            elif lower.endswith(".docx"):
                file_type = "docx"
                doc = docx.Document(temp_path)
                text = "\n".join([p.text.strip() for p in doc.paragraphs]).strip()
            elif lower.endswith((".xlsx", ".xls")):
                file_type = "excel"
                df = pd.read_excel(temp_path)
                text = "\n".join([str(df)]).strip()
            elif lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                file_type = "image"
                try:
                    if _HAS_EASYOCR:
                        # Initialize easyocr reader here to avoid torch initialization issues
                        reader = easyocr.Reader(["en"], gpu=(_HAS_TORCH and torch.cuda.is_available()))
                        results = reader.readtext(temp_path, detail=0)
                        text = "\n".join(results).strip()
                    else:
                        text = pytesseract.image_to_string(Image.open(temp_path)).strip()
                except Exception:
                    # Fallback to pytesseract if easyocr or initial attempt fails
                    text = pytesseract.image_to_string(Image.open(temp_path)).strip()
            elif lower.endswith(".txt"):
                file_type = "txt"
                with open(temp_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            else:
                raise ValueError(f"Unsupported file type: {filename}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        return text, file_type

    # --- Text Utilities (Combined) ---
    def simple_sent_tokenize(self, text: str) -> List[str]:
        # Refined tokenization
        text = re.sub(r'\s+', ' ', text.replace("\n", " ").strip())
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]

    def chunk_sentences_by_wordcount(self, sentences: List[str], max_words: int) -> List[str]:
        # Robust chunking logic
        chunks, curr, curr_count = [], [], 0
        for s in sentences:
            w = len(s.split())
            if curr_count + w > max_words and curr:
                chunks.append(" ".join(curr).strip())
                curr = [s]
                curr_count = w
            else:
                curr.append(s)
                curr_count += w
        if curr:
            chunks.append(" ".join(curr).strip())
        return chunks

    # --- Knowledge Base (RAG) ---
    def build_knowledge_base(self, text):
        if not text:
            raise ValueError("Document text is empty or could not be extracted.")
            
        self.full_text = text
        sentences = self.simple_sent_tokenize(text)
        # Use the larger RAG chunk size
        self.knowledge_chunks = self.chunk_sentences_by_wordcount(sentences, max_words=MAX_CHUNK_WORDS)
        
        if not self.knowledge_chunks:
            raise ValueError("Document text was extracted but no valid knowledge chunks could be formed.")

        print(f"Encoding {len(self.knowledge_chunks)} chunks...")
        embeddings = self.embed_model.encode(self.knowledge_chunks, convert_to_tensor=False)
        
        dim = embeddings[0].shape[0]
        self.knowledge_index = faiss.IndexFlatL2(dim)
        self.knowledge_index.add(np.array(embeddings).astype("float32"))
        
        return len(self.knowledge_chunks)

    # --- Document Review Functions (From Gradio) ---

    def extract_key_sentences(self, text: str, top_k: int = TOP_K_SENT_EXTRACT) -> List[str]:
        """Extracts key sentences by measuring similarity to the document centroid."""
        if not text.strip():
            return []
        sentences = self.simple_sent_tokenize(text)
        
        # Ensure sentences are not too short to form meaningful embeddings
        # Ensure tensor is moved to CPU for util.cos_sim if using a complex setup
        sent_embeddings = self.embed_model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        # Move to CPU only if this is a PyTorch tensor
        if hasattr(sent_embeddings, "to"):
            sent_embeddings = sent_embeddings.to('cpu')
            centroid = sent_embeddings.mean(dim=0, keepdim=True)
        else:
            # numpy fallback
            centroid = np.mean(sent_embeddings, axis=0, keepdims=True)
        scores = util.cos_sim(sent_embeddings, centroid).squeeze().tolist()
            
        # Ensure tensor is moved to CPU for util.cos_sim if using a complex setup
        sent_embeddings = self.embed_model.encode(sentences, convert_to_tensor=True, show_progress_bar=False).to('cpu')
        centroid = sent_embeddings.mean(dim=0, keepdim=True)
        scores = util.cos_sim(sent_embeddings, centroid).squeeze().tolist()
        
        scored = list(zip(sentences, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by similarity threshold
        filtered = [s for s, sc in scored if sc >= SIMILARITY_THRESHOLD_SELECT]
        
        # If filtering is too aggressive, revert to a few top sentences
        if not filtered:
             filtered = [s for s, sc in scored[:min(5, len(scored))]]

        # Deduplicate (remove near-identical sentences)
        seen = set()
        key_points = []
        for kp in filtered:
            norm = kp.lower().strip()
            if norm not in seen:
                seen.add(norm)
                key_points.append(kp)

        return key_points[:min(top_k, len(key_points))]

    def lint_suggestions(self, text: str) -> List[str]:
        """Provides quality suggestions for legal/general documents."""
        suggestions = []
        lc = text.lower()
        
        # Mandatory checks
        if re.search(r"\b(date|effective date)\b", lc) is None:
            suggestions.append("No explicit 'date' detected.")
        if re.search(r"\b(party|parties|company|vendor|client|supplier)\b", lc) is None:
            suggestions.append("No parties clearly identified.")
            
        # Rule-based checks
        for pat, msg in SUGGESTION_RULES:
            if re.search(pat, lc):
                suggestions.append(msg)
                
        # Length check
        long_sentences = [s for s in self.simple_sent_tokenize(text) if len(s.split()) > 60]
        if long_sentences:
            suggestions.append(f"{len(long_sentences)} very long sentence(s) detected. Consider splitting for clarity.")
            
        return list(dict.fromkeys(suggestions))

    # --- Summarization (Refined from RAG/Gradio) ---
    def summarize_document(self):
        if not self.full_text:
            return "No document is currently loaded for summarization."
        
        full_text = self.full_text
        
        # Tokenize and truncate safely
        token_ids = self.summ_tokenizer.encode(full_text, truncation=False, return_tensors='pt')[0]
        
        if len(token_ids) > T5_MAX_INPUT_LENGTH:
            token_ids = token_ids[:T5_MAX_INPUT_LENGTH]
            input_text = self.summ_tokenizer.decode(token_ids, skip_special_tokens=True)
            print(f"Warning: Document text truncated to {T5_MAX_INPUT_LENGTH} tokens for summarization.")
        else:
            input_text = full_text
            
        # Prepend the T5 instruction
        input_text = "summarize: " + input_text
            
        try:
            summary_result = self.summarizer_pipeline(
                input_text,
                max_length=500,
                min_length=30,
                do_sample=False
            )
            return summary_result[0]['summary_text']
        except Exception as e:
            print(f"Summarization error: {e}")
            return "An error occurred during summarization. The document may be too large or the model failed."

    # --- Safety Check (From RAG) ---
    def is_safe_model(self, prompt):
        results = self.safety_pipeline(prompt, top_k=1)
        if results and results[0]['label'].lower() == 'toxic' and results[0]['score'] > 0.8:
            return False
        return True

    # --- Web Search (Trusted, From RAG) ---
    def search_trusted_web(self, query, domain=None, num_results=5):
        url = f"https://serpapi.com/search.json"
        
        search_query = f"{query}" if not domain else f"site:{domain} {query}"
        
        params = {
            "engine": "google",
            "q": search_query,
            "api_key": self.SERPAPI_KEY,
            "num": num_results,
            "gl": "us",
            "hl": "en"
        }
        
        try:
            res = requests.get(url, params=params, timeout=10).json()
            texts = []
            if 'error' in res:
                print(f"SerpAPI Error: {res['error']}")
                return ["Warning: Web search failed due to API error. Relying only on local context."]
                
            for r in res.get("organic_results", []):
                snippet = r.get("snippet", "")
                texts.append(snippet)
            return texts
        except requests.RequestException as e:
            print(f"SerpAPI request failed: {e}")
            return ["Warning: Web search failed due to network error. Relying only on local context."]

    # --- RAG Query Handler (Combined/Refined) ---
    def query_rag_web(self, question):
        if not self.is_safe_model(question):
            return "⚠ Unsafe query blocked. The content violates safety guidelines."
            
        # 1. Local Retrieval (RAG)
        local_chunks = []
        if self.knowledge_index is not None:
            try:
                q_emb = self.embed_model.encode([question], convert_to_tensor=False)
                D, I = self.knowledge_index.search(np.array(q_emb).astype("float32"), TOP_K_RAG)
                valid_indices = [i for i in I[0] if i != -1 and i < len(self.knowledge_chunks)]
                local_chunks = [self.knowledge_chunks[i] for i in valid_indices]
            except Exception as e:
                print(f"Local retrieval error: {e}")

        # 2. Web Search Category Detection
        q_lower = question.lower()
        trusted_domain = None
        for cat_name, cat_data in TRUSTED_DOMAINS.items():
            if any(keyword in q_lower for keyword in cat_data['keywords']):
                trusted_domain = cat_data['domain']
                print(f"Detected category '{cat_name}'. Searching trusted domain: {trusted_domain}")
                break
                
        web_chunks = self.search_trusted_web(question, trusted_domain)

        # 3. Combine Context and Answer Generation
        context = " ".join(local_chunks + web_chunks)
        
        if not context.strip():
            return "No relevant context was retrieved from either the uploaded document or the web search to answer the question."

        try:
            # Use the robust QA pipeline
            out = self.qa_pipeline(
                question=question, 
                context=context,
                handle_impossible_answer=True
            )
            
            answer = out.get('answer', 'No specific answer found in the provided context.')
            
            # Format the output with confidence score (from Gradio's QA function)
            confidence = out.get('score', 0.0)
            formatted_answer = f"Answer ({confidence:.2f}): {answer}"
            
        except Exception as e:
            print(f"QA pipeline error: {e}")
            formatted_answer = "Unable to generate answer from the combined context."
            
        # 4. Log and Return
        self.log_query(question, formatted_answer)
        return formatted_answer

# -----------------------------
# FLASK API SETUP
# -----------------------------
app = Flask(__name__)
CORS(app) 

# Initialize the Chatbot instance globally
print("\n--- Initializing RAG Chatbot ---")
try:
    chatbot = RAGChatbot(serpapi_key=SERPAPI_KEY, db_file=DB_FILE)
except Exception as e:
    print(f"Failed to initialize chatbot: {e}")
    raise

# --- FLASK ROUTES ---

@app.route("/")
def serve_frontend():
    return render_template_string(HTML_CONTENT)

@app.route("/upload", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    
    file = request.files["file"]
    
    try:
        # Load and set full_text
        text_content, file_type = chatbot.load_document(file)
        # Build KB
        chunk_count = chatbot.build_knowledge_base(text_content)
        
        return jsonify({
            "status": "success", 
            "message": f"Document processed ({file_type}) and knowledge base built with {chunk_count} chunks.",
            "full_text": text_content # Return full text for client-side context management
        }), 200
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        print(f"Processing error: {e}")
        return jsonify({"status": "error", "message": "An error occurred during document processing. Check the console for details."}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "")
    
    if not question:
        return jsonify({"answer": "Please provide a question."}), 400
        
    try:
        # Use the combined RAG/Web Query handler
        answer = chatbot.query_rag_web(question)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        print(f"Query error: {e}")
        return jsonify({"answer": "An unexpected error occurred during the query. Please check server logs."}), 500

@app.route("/summarize", methods=["POST"])
def summarize_document_api():
    try:
        summary = chatbot.summarize_document()
        if summary.startswith("No document") or "error occurred during summarization" in summary:
             return jsonify({"summary": summary, "status": "error", "message": summary}), 400
        return jsonify({"summary": summary, "status": "success"}), 200
    except Exception as e:
        print(f"Summarize API error: {e}")
        return jsonify({"summary": "An unexpected error occurred during summarization.", "status": "error"}, 500)

@app.route("/keypoints", methods=["POST"])
def get_keypoints_api():
    # Assumes the full_text state in the chatbot is up-to-date from the last /upload
    if not chatbot.full_text:
        return jsonify({"keypoints": "Please upload a document first.", "status": "error"}), 400
    try:
        keypoints = chatbot.extract_key_sentences(chatbot.full_text)
        return jsonify({"keypoints": "\n".join(keypoints), "status": "success"}), 200
    except Exception as e:
        print(f"Keypoints API error: {e}")
        return jsonify({"keypoints": "An error occurred during key sentence extraction.", "status": "error"}, 500)

@app.route("/suggestions", methods=["POST"])
def get_suggestions_api():
    # Assumes the full_text state in the chatbot is up-to-date from the last /upload
    if not chatbot.full_text:
        return jsonify({"suggestions": "Please upload a document first.", "status": "error"}), 400
    try:
        suggestions = chatbot.lint_suggestions(chatbot.full_text)
        return jsonify({"suggestions": "\n".join(suggestions), "status": "success"}), 200
    except Exception as e:
        print(f"Suggestions API error: {e}")
        return jsonify({"suggestions": "An error occurred during suggestion generation.", "status": "error"}, 500)

# -----------------------------
# EMBEDDED HTML/CSS/JS FRONTEND
# -----------------------------
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Document RAG & Review Assistant 🧠</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Custom scrollbar for chat area */
        .custom-scrollbar::-webkit-scrollbar {
            width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
            background-color: #3b82f6; /* primary */
            border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
            background-color: #f3f4f6; /* gray-100 */
        }
        /* Typing animation dots */
        .typing span {
            animation: blink 1s infinite;
        }
        .typing span:nth-child(2) {
            animation-delay: 0.2s;
        }
        .typing span:nth-child(3) {
            animation-delay: 0.4s;
        }
        @keyframes blink {
            0%, 100% { opacity: 0.2; }
            50% { opacity: 1; }
        }
        /* Active Tab Styling */
        .tab-button.active {
            border-bottom: 3px solid #10b981; /* secondary color */
            font-weight: 600;
            color: #10b981;
        }
        /* New: Floating Chat Container Styling */
        #floatingChatContainer {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }
        #chatWindow {
            height: 50vh; /* Adjust height for the widget */
            min-height: 400px;
        }
        /* New: Hide/Show chat window transition */
        .chat-hidden {
            transform: translateY(100%) scale(0.9);
            opacity: 0;
            pointer-events: none;
        }
        .chat-visible {
            transform: translateY(0) scale(1);
            opacity: 1;
            pointer-events: auto;
        }
    </style>
</head>
<body class="bg-gray-50 font-sans antialiased h-screen flex flex-col">
    <script>
        // Set Tailwind config 
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                    },
                    colors: {
                        'primary': '#3b82f6', // Blue-500
                        'secondary': '#10b981', // Emerald-500
                    }
                }
            }
        }
    </script>

    <div class="flex flex-1 overflow-hidden">
        
        <aside class="flex-1 bg-white p-6 shadow-2xl flex flex-col space-y-6 overflow-y-auto border-r border-gray-200">
            <h2 class="text-3xl font-extrabold text-primary border-b-4 border-primary/20 pb-3">📑 Document Review</h2>

            <div class="bg-blue-50 p-5 rounded-xl border border-blue-200">
                <h3 class="font-bold text-xl text-gray-800 mb-3 flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-6 h-6 mr-2 text-primary">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.75 3.75 0 0118 19.5H6.75z" />
                    </svg>
                    Upload & Process
                </h3>
                <input type="file" id="fileInput" class="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-white focus:ring-primary focus:border-primary p-2.5">
                <button onclick="handleUpload()" id="uploadButton" class="w-full mt-4 px-4 py-2 bg-primary text-white font-bold rounded-lg hover:bg-blue-600 transition duration-150 shadow-md flex items-center justify-center">
                    <span id="uploadText">Process Document & Build KB</span>
                </button>
            </div>
            
            <div class="flex border-b border-gray-200">
                <button class="tab-button active p-3 text-lg transition duration-150 ease-in-out" onclick="changeTab('summary')">📝 Summary</button>
                <button class="tab-button p-3 text-lg transition duration-150 ease-in-out" onclick="changeTab('keypoints')">🔑 Key Points</button>
                <button class="tab-button p-3 text-lg transition duration-150 ease-in-out" onclick="changeTab('suggestions')">💡 Suggestions</button>
            </div>
            
            <div id="reviewContent" class="flex-1 overflow-y-auto custom-scrollbar space-y-4">
                <div id="summary" class="tab-pane">
                    <textarea id="summaryOutput" rows="10" class="w-full p-3 border border-gray-300 rounded-lg bg-gray-50 text-gray-700" readonly placeholder="Summary will appear here..."></textarea>
                </div>
                <div id="keypoints" class="tab-pane hidden">
                    <textarea id="keypointsOutput" rows="10" class="w-full p-3 border border-gray-300 rounded-lg bg-gray-50 text-gray-700" readonly placeholder="Key points will appear here..."></textarea>
                </div>
                <div id="suggestions" class="tab-pane hidden">
                    <textarea id="suggestionsOutput" rows="10" class="w-full p-3 border border-gray-300 rounded-lg bg-gray-50 text-gray-700" readonly placeholder="Document suggestions/linting results will appear here..."></textarea>
                </div>
            </div>

        </aside>

        </div>

    <div id="floatingChatContainer">
        <button id="chatBubble" onclick="toggleChat()" class="bg-secondary text-white p-4 rounded-full shadow-2xl hover:bg-emerald-600 transition duration-300 transform hover:scale-105">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-8 h-8">
                <path fill-rule="evenodd" d="M4.848 2.771A9.75 9.75 0 0112 2.25c5.385 0 9.75 4.365 9.75 9.75s-4.365 9.75-9.75 9.75c-1.383 0-2.75-.356-4.004-1.018a.75.75 0 00-.707.025L4.04 21.423a.75.75 0 01-1.04-.216l-.805-1.423a.75.75 0 01.328-.975 9.47 9.47 0 01-1.02-.75.75.75 0 00-.003-1.05z" clip-rule="evenodd" />
            </svg>
        </button>

        <div id="chatWidget" class="absolute bottom-0 right-0 transform transition-all duration-300 shadow-2xl bg-white rounded-xl overflow-hidden w-96 flex flex-col chat-hidden">
            
            <div class="bg-secondary text-white p-3 flex justify-between items-center">
                <h3 class="font-bold">🧠 RAG Assistant</h3>
                <button onclick="toggleChat()" class="hover:bg-emerald-600 p-1 rounded">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2.5" stroke="currentColor" class="w-5 h-5">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>

            <div id="chatWindow" class="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4 bg-gray-50">
                <div class="flex justify-start">
                    <div class="bg-gray-200 text-gray-800 p-3 rounded-xl rounded-tl-none max-w-xs text-sm shadow-sm">
                        👋 **RAG Ready!** Ask questions about the document or general topics.
                    </div>
                </div>
            </div>
            
            <form id="queryFormFloating" class="flex p-3 border-t border-gray-200">
                <input type="text" id="queryInputFloating" placeholder="Ask a question..." class="flex-1 p-2 border border-gray-300 rounded-l-lg focus:ring-secondary focus:border-secondary transition duration-150 text-sm">
                <button type="submit" id="sendButtonFloating" class="bg-secondary text-white p-2 rounded-r-lg font-bold hover:bg-emerald-600 transition duration-150 flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                        <path d="M3.478 2.405a.75.75 0 01.442.822l-1.355 7.135A.75.75 0 003.41 12l13.59 1.942a.75.75 0 01.66.721v.172l-1.355 7.135a.75.75 0 01-1.424.385L8.985 15.659a.75.75 0 00-.73-.557l-1.355 7.135a.75.75 0 01-1.424.385l-.566-2.973H3.97a.75.75 0 01-.75-.75V11.25c0-.414.336-.75.75-.75h.005l.566-2.973a.75.75 0 011.424-.385l5.222 2.748a.75.75 0 00.73.557l1.355-7.135a.75.75 0 01.442-.822z" />
                    </svg>
                </button>
            </form>
        </div>
    </div>

    <script>
        const chatWidget = document.getElementById('chatWidget');
        const chatBubble = document.getElementById('chatBubble');
        // NOTE: Must select the specific chatWindow within the widget to avoid conflict
        const chatWindow = chatWidget.querySelector('#chatWindow'); 
        const queryForm = document.getElementById('queryFormFloating');
        const queryInput = document.getElementById('queryInputFloating');
        const uploadButton = document.getElementById('uploadButton');
        const uploadText = document.getElementById('uploadText');
        
        let isProcessing = false;
        let isChatOpen = false;

        function toggleChat() {
            isChatOpen = !isChatOpen;
            if (isChatOpen) {
                chatWidget.classList.remove('chat-hidden');
                chatWidget.classList.add('chat-visible');
                chatBubble.classList.add('hidden'); 
            } else {
                chatWidget.classList.remove('chat-visible');
                chatWidget.classList.add('chat-hidden');
                chatBubble.classList.remove('hidden');
            }
        }

        function addMessage(text, isUser) {
            const container = document.createElement('div');
            container.className = 'flex ' + (isUser ? 'justify-end' : 'justify-start');
            
            const bubble = document.createElement('div');
            bubble.className = 'p-3 rounded-xl max-w-xs text-sm shadow-md break-words ' + 
                               (isUser ? 'bg-primary text-white rounded-br-none' : 'bg-gray-200 text-gray-800 rounded-tl-none');
            bubble.innerHTML = text;
            
            container.appendChild(bubble);
            chatWindow.appendChild(container);
            chatWindow.scrollTop = chatWindow.scrollHeight;
            return bubble;
        }

        function showTyping() {
            const typingMsg = document.createElement('div');
            typingMsg.className = 'flex justify-start';
            typingMsg.id = 'typingIndicator';
            typingMsg.innerHTML = `
                <div class="bg-gray-200 text-gray-800 p-3 rounded-xl rounded-tl-none max-w-xs text-sm shadow-sm typing">
                    Assistant is thinking<span>.</span><span>.</span><span>.</span>
                </div>
            `;
            chatWindow.appendChild(typingMsg);
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }

        function hideTyping() {
            const indicator = document.getElementById('typingIndicator');
            if (indicator) {
                indicator.remove();
            }
        }
        
        function changeTab(tabId) {
            document.querySelectorAll('.tab-pane').forEach(pane => {
                pane.classList.add('hidden');
            });
            document.getElementById(tabId).classList.remove('hidden');

            document.querySelectorAll('.tab-button').forEach(button => {
                button.classList.remove('active');
            });
            document.querySelector(`.tab-button[onclick*='${tabId}']`).classList.add('active');
        }

        async function fetchReviewData(endpoint, outputId) {
            const outputElement = document.getElementById(outputId);
            outputElement.value = "Processing...";

            try {
                const response = await fetch(endpoint, { method: 'POST' });
                const data = await response.json();
                
                if (data.status === 'success') {
                    outputElement.value = data[outputId.replace('Output', '').toLowerCase()];
                } else {
                    outputElement.value = `Error: ${data.message || 'Could not retrieve data. Please check if a document is loaded.'}`;
                }
            } catch (error) {
                console.error(`Error fetching ${endpoint}:`, error);
                outputElement.value = "A network or server error occurred. Check the console.";
            }
        }
        
        async function handleUpload() {
            if (isProcessing) return;
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert("Please select a file to upload.");
                return;
            }

            isProcessing = true;
            uploadButton.disabled = true;
            uploadText.innerText = "Processing... (This may take a minute)";
            
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.status === 'success') {
                    addMessage(`✅ **Document Loaded:** ${data.message}. You can now start chatting!`, false);
                    
                    // Automatically run and display document review results
                    await Promise.all([
                        fetchReviewData('/summarize', 'summaryOutput'),
                        fetchReviewData('/keypoints', 'keypointsOutput'),
                        fetchReviewData('/suggestions', 'suggestionsOutput')
                    ]);
                    changeTab('summary'); // Show summary first
                    
                } else {
                    addMessage(`❌ **Upload Failed:** ${data.message}`, false);
                    document.getElementById('summaryOutput').value = data.message;
                    document.getElementById('keypointsOutput').value = data.message;
                    document.getElementById('suggestionsOutput').value = data.message;
                }
            } catch (error) {
                console.error('Upload error:', error);
                addMessage(`❌ **Critical Upload Error:** Failed to connect to server.`, false);
                document.getElementById('summaryOutput').value = "Critical server error.";
            } finally {
                isProcessing = false;
                uploadButton.disabled = false;
                uploadText.innerText = "Process Document & Build KB";
            }
        }

        queryForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const question = queryInput.value.trim();
            if (!question) return;

            addMessage(question, true);
            queryInput.value = '';
            
            showTyping();

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });

                const data = await response.json();
                hideTyping();
                addMessage(data.answer, false);

            } catch (error) {
                hideTyping();
                addMessage("An error occurred while trying to get an answer.", false);
                console.error('Query error:', error);
            }
        });
        
        // Initialize the first tab view
        document.addEventListener('DOMContentLoaded', () => {
            changeTab('summary');
        });

    </script>
</body>
</html>
"""

if __name__ == "__main__":
    # Ensure all models are loaded before starting the server
    print("\n--- Starting Flask Server ---")
    app.run(debug=False, host='0.0.0.0', port=5000)
