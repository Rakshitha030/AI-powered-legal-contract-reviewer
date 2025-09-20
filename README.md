# AI-powered-legal-contract-reviewer

This project is an AI-powered legal document assistant that can:

📤 Extract text from PDF, DOCX, Images (OCR), TXT

📝 Summarize long contracts into concise plain English

🔑 Extract key points & important clauses

💡 Provide suggestions to improve clarity, enforceability, and detect missing details

❓ Answer questions about the contract (Q&A system)

🖥️ Interactive Gradio UI for easy usage


## ✨ Features

✅ Multi-format Support → PDF, DOCX, TXT, PNG/JPG (OCR via Tesseract/EasyOCR)

✅ Hybrid Summarization → Uses embeddings + abstractive summarization (T5-small)

✅ Key Sentence Extraction → Finds the most important sentences using sentence-transformers

✅ Legal Suggestions → Detects missing clauses, weak obligations, unclear deadlines

✅ Question Answering → Ask contract-specific questions (DistilBERT QA model)

✅ Language Detection → Auto-detects document language (if supported)

✅ User-Friendly UI → Gradio-based interface

## 🛠️ Tech Stack

Backend / Core: Python

Libraries:

pdfplumber, python-docx, PIL, pytesseract, easyocr (OCR & document parsing)

transformers, sentence-transformers (summarization, embeddings, Q&A)

langdetect (language detection)

gradio (interactive web app)

Models:

Summarization → t5-small

Q&A → distilbert-base-uncased-distilled-squad

Embeddings → paraphrase-MiniLM-L3-v2




## ⚖️ Workflow Diagram

```mermaid
flowchart TD
    A[Document Upload - PDF, DOCX, TXT, Image ] --> B[Text Extraction - pdfplumber, docx, pytesseract, easyOCR]
    B --> C[Preprocessing - Cleaning, Language Detection, Tokenizing]
    C --> D1[Summarization - T5-small]
    C --> D2[Key Points Extraction - Sentence Embeddings]
    C --> D3[Suggestions Generator - Rule-based ]
    C --> D4[Q&A System - DistilBERT]
    D1 --> E[Results Display - Summary, Keypoints, Suggestions, Q&A]
    D2 --> E
    D3 --> E
    D4 --> E
'''

