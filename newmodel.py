import os
import re
from typing import List, Tuple

import pdfplumber
import docx
from PIL import Image
import pytesseract

try:
    import easyocr
    _HAS_EASYOCR = True
except Exception:
    _HAS_EASYOCR = False

import gradio as gr
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

try:
    from langdetect import detect
    _HAS_LANGDETECT = True
except Exception:
    _HAS_LANGDETECT = False

# ---------------------------
# Configuration / Model names
# ---------------------------
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"   # tiny
SUMMARIZER_MODEL = "t5-small"
QA_MODEL = "distilbert-base-uncased-distilled-squad"


SIMILARITY_THRESHOLD_ANNOTATE = 0.62
SIMILARITY_THRESHOLD_SELECT = 0.52
MAX_CHUNK_WORDS = 300

# ---------------------------
# Device / Memory config
# ---------------------------
device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ---------------------------
# Load models
# ---------------------------
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL, device='cuda' if torch.cuda.is_available() else 'cpu')

print("Loading summarizer tokenizer & model...")
summ_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL, legacy=False, use_fast=True)
summ_model = AutoModelForSeq2SeqLM.from_pretrained(
    SUMMARIZER_MODEL,
    device_map="auto" if device >= 0 else None,
    dtype=dtype,
)
summarizer_pipeline = pipeline(
    "summarization",
    model=summ_model,
    tokenizer=summ_tokenizer,
    framework="pt",
    device=device,
    max_length=200,
    min_length=30,
    do_sample=False,
)

print("Loading QA pipeline...")
qa_pipeline = pipeline(
    "question-answering",
    model=QA_MODEL,
    tokenizer=QA_MODEL,
    device=device,
)

# ---------------------------
# OCR helpers
# ---------------------------
def ocr_image_pytesseract(image_path: str, lang: str = "eng") -> str:
    return pytesseract.image_to_string(Image.open(image_path), lang=lang)

def ocr_image_easyocr(image_path: str, langs=None) -> str:
    if not _HAS_EASYOCR:
        raise RuntimeError("easyocr not installed")
    reader = easyocr.Reader(langs if langs else ["en"], gpu=torch.cuda.is_available())
    results = reader.readtext(image_path, detail=0)
    return "\n".join(results)

# ---------------------------
# Document loading
# ---------------------------
def load_document(file_obj) -> Tuple[str, str]:
    path = getattr(file_obj, "name", None) or getattr(file_obj, "temp_path", None)
    if not path:
        raise ValueError("Cannot read uploaded file")
    path = str(path)
    lower = path.lower()
    if lower.endswith(".pdf"):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text.strip(), "pdf"
    elif lower.endswith(".docx"):
        doc = docx.Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text.strip(), "docx"
    elif any(lower.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")):
        try:
            if _HAS_EASYOCR:
                txt = ocr_image_easyocr(path)
            else:
                txt = ocr_image_pytesseract(path)
        except Exception:
            txt = ocr_image_pytesseract(path)
        return txt.strip(), "image"
    elif lower.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip(), "txt"
    else:
        raise ValueError("Unsupported file type.")

# ---------------------------
# Text utils
# ---------------------------
def simple_sent_tokenize(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', text.replace("\n", " ").strip())
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def chunk_sentences_by_wordcount(sentences: List[str], max_words: int) -> List[str]:
    chunks, curr, curr_count = [], [], 0
    for s in sentences:
        w = len(s.split())
        if curr_count + w > max_words and curr:
            chunks.append(" ".join(curr))
            curr = [s]
            curr_count = w
        else:
            curr.append(s)
            curr_count += w
    if curr:
        chunks.append(" ".join(curr))
    return chunks

# ---------------------------
# Core summarization
# ---------------------------
def extract_key_sentences(text: str, top_k: int = 8) -> List[Tuple[str, float]]:
    if not text.strip():
        return []
    sentences = simple_sent_tokenize(text)
    sent_embeddings = embed_model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
    centroid = sent_embeddings.mean(dim=0, keepdim=True)
    scores = util.cos_sim(sent_embeddings, centroid).squeeze().tolist()
    scored = list(zip(sentences, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:min(top_k, len(scored))]

def hybrid_summarize(text: str, max_summary_words: int = 120) -> str:
    candidates = extract_key_sentences(text, top_k=20)
    filtered = [s for s, sc in candidates if sc >= SIMILARITY_THRESHOLD_SELECT]
    if not filtered:
        filtered = [s for s, sc in candidates[:6]]
    chunks = chunk_sentences_by_wordcount(filtered, MAX_CHUNK_WORDS)
    summaries = []
    for chunk in chunks:
        try:
            out = summarizer_pipeline("summarize: " + chunk, max_length=max_summary_words)
            summaries.append(out[0]["summary_text"])
        except RuntimeError as e:
            print(f"[Memory issue during summarization: {e}]")
            summaries.append(chunk)
    return " ".join(summaries).strip()

# ---------------------------
# Annotation & QA
# ---------------------------
def annotate_text(text: str, summary: str) -> Tuple[str, List[str]]:
    sentences = simple_sent_tokenize(text)
    if not sentences:
        return "", []
    summary_points = simple_sent_tokenize(summary) or [s for s,_ in extract_key_sentences(text, top_k=6)]
    sent_emb = embed_model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
    sum_emb = embed_model.encode(summary_points, convert_to_tensor=True, show_progress_bar=False)
    annotated, key_sents = [], []
    for i, s in enumerate(sentences):
        sims = util.cos_sim(sent_emb[i], sum_emb).squeeze().tolist()
        if not isinstance(sims, list):
            sims = [sims]
        max_sim = max(sims) if sims else 0.0
        if max_sim >= SIMILARITY_THRESHOLD_ANNOTATE:
            annotated.append(f"[KEY_POINT] {s}")
            key_sents.append(s)
        else:
            annotated.append(s)
    return " ".join(annotated), key_sents

def answer_question(question: str, context: str) -> str:
    if not context or not question.strip():
        return "No context or question provided."
    try:
        out = qa_pipeline(question=question, context=context)
        return f"Answer ({out.get('score',0.0):.2f}): {out.get('answer','')}"
    except Exception as e:
        return f"[QA Error] {e}"

# ---------------------------
# Suggestions
# ---------------------------
SUGGESTION_RULES = [
    (r"\b(may|could|might|should)\b", "Consider replacing weak modal verbs with clearer obligations."),
    (r"\b(payment|amount|fee|rupees|\$|\€|\£)\b", "Ensure currency and payment schedule are clear."),
    (r"\b(deadline|due date|within \d+ (days|weeks|months))\b", "Add unambiguous deadlines."),
    (r"\b(terminate|termination)\b", "Check termination clauses."),
    (r"\b(liabilit(y|ies)|indemnif)\b", "Ensure liability/indemnity caps and exclusions are explicit."),
]

def lint_suggestions(text: str) -> List[str]:
    suggestions = []
    lc = text.lower()
    if re.search(r"\b(date|effective date)\b", lc) is None:
        suggestions.append("No explicit 'date' detected.")
    if re.search(r"\b(party|parties|company|vendor|client|supplier)\b", lc) is None:
        suggestions.append("No parties clearly identified.")
    for pat, msg in SUGGESTION_RULES:
        if re.search(pat, lc) and msg:
            suggestions.append(msg)
    long_sentences = [s for s in simple_sent_tokenize(text) if len(s.split()) > 60]
    if long_sentences:
        suggestions.append(f"{len(long_sentences)} very long sentence(s) detected.")
    return list(dict.fromkeys(suggestions))

# ---------------------------
# Full pipeline
# ---------------------------
def process_document_file(file_obj):
    try:
        raw_text, source = load_document(file_obj)
    except Exception as e:
        return {"error": f"Failed to load document: {e}", "summary": "", "annotations": "", "key_sentences": [], "detected_questions": [], "suggestions": []}
    if not raw_text.strip():
        return {"error": "No text extracted.", "summary": "", "annotations": "", "key_sentences": [], "detected_questions": [], "suggestions": []}

    detected_lang = None
    if _HAS_LANGDETECT:
        try:
            detected_lang = detect(raw_text[:2000])
        except Exception:
            detected_lang = None

    try:
        summary = hybrid_summarize(raw_text)
    except Exception as e:
        summary = f"[Summary generation failed: {e}]"

    except Exception:
        annotated_text, key_sents = raw_text, []

    detected_questions = [s for s in simple_sent_tokenize(raw_text) if s.strip().endswith("?")]
    suggestions = lint_suggestions(raw_text)

    return {
        "error": None,
        "source": source,
        "detected_language": detected_lang,
        "summary": summary,
        "annotated_text": annotated_text,
        "key_sentences": key_sents,
        "detected_questions": detected_questions,
        "suggestions": suggestions,
    }

# ---------------------------
# Gradio UI
# ---------------------------
# ---------------------------
# Gradio UI
# ---------------------------
def gr_process_file(file_obj):
    text, ftype = load_document(file_obj)
    summary = hybrid_summarize(text)

    # Get top key sentences
    key_points_raw = [s for s, _ in extract_key_sentences(text, top_k=12)]

    # Deduplicate (remove near-identical sentences)
    seen = set()
    key_points = []
    for kp in key_points_raw:
        norm = kp.lower().strip()
        if norm not in seen:
            seen.add(norm)
            key_points.append(kp)

    suggestions = lint_suggestions(text)
    return summary, "\n".join(key_points[:8]), "\n".join(suggestions), text



def gr_answer_question(question, context):
    return answer_question(question, context)


with gr.Blocks(title="AI Legal Contract Reviewer") as demo:
    gr.Markdown("## 📑 AI Legal Document Reviewer")

    with gr.Row():
        file_input = gr.File(label="Upload a document", type="filepath")
        proc_btn = gr.Button("Process Document")

    summary_output = gr.Textbox(label="📝 Summary", lines=8, interactive=False)
    keypoints_output = gr.Textbox(label="🔑 Key Points", lines=8, interactive=False)
    suggestions_output = gr.Textbox(label="💡 Suggestions", lines=8, interactive=False)

    with gr.Row():
        question_input = gr.Textbox(label="Ask a Question", placeholder="Type your question here...")
        ask_btn = gr.Button("Ask")

    answer_output = gr.Markdown(label="💡 Answer")

    # Hidden state to store raw text for QA
    context_state = gr.State("")

    proc_btn.click(
        fn=gr_process_file,
        inputs=[file_input],
        outputs=[summary_output, keypoints_output, suggestions_output, context_state],
    )

    ask_btn.click(
        fn=gr_answer_question,
        inputs=[question_input, context_state],
        outputs=[answer_output],
    )

if __name__ == "__main__":
    demo.launch(share=True)
