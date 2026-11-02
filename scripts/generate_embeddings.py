"""
Embedding and FAISS Index Generation Script.

This script:
1. Generates or loads a synthetic enterprise corpus of at least 1,000 documents across
   domains: Technology, Healthcare, Finance, and E-commerce.
2. Loads a pre-trained SentenceTransformer model (default: sentence-transformers/all-MiniLM-L6-v2).
3. Computes dense vector embeddings for all documents.
4. Caches the model locally in models/sentence_transformer/.
5. Builds a FAISS IndexFlatL2 index.
6. Persists data/documents.json, data/embeddings.npy, and data/faiss_index.bin.
7. Executes a verification search sanity check.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("generate_embeddings")

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path_str: str) -> Path:
    """Resolve path: if absolute, return as-is; otherwise anchor to PROJECT_ROOT."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", str(PROJECT_ROOT / "models" / "sentence_transformer"))
DEFAULT_DATA_PATH = os.getenv("DATA_PATH", str(PROJECT_ROOT / "data" / "documents.json"))
DEFAULT_FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", str(PROJECT_ROOT / "data" / "faiss_index.bin"))
DEFAULT_EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", str(PROJECT_ROOT / "data" / "embeddings.npy"))


def generate_synthetic_documents(count: int = 1200) -> List[Dict[str, str]]:
    """
    Generate high quality, diverse synthetic documents across four key industries:
    Technology, Healthcare, Finance, and E-commerce.
    """
    logger.info(f"Generating {count} synthetic domain documents...")

    technologies = [
        "Kubernetes container orchestration automates deployment, scaling, and operational management of containerized applications.",
        "Vector databases enable low-latency approximate nearest neighbor search across billions of high-dimensional embeddings.",
        "FastAPI provides modern, high-performance asynchronous web APIs with automatic OpenAPI and JSON Schema generation.",
        "CI/CD automated testing pipelines ensure microservice stability and streamline continuous deployment across staging environments.",
        "Microservices architecture decomposes monolithic backends into loosely coupled, independently scalable domain services.",
        "Transformer architecture leverages multi-head self-attention mechanisms to capture contextual dependencies in natural language.",
        "Zero-trust network architecture requires continuous identity authentication and cryptographic validation for all service requests.",
        "Event-driven streaming with Apache Kafka enables real-time distributed telemetry processing and analytics at enterprise scale.",
        "Retrieval-Augmented Generation (RAG) augments large language models with dynamic vector database retrieval to prevent hallucinations.",
        "Docker multi-stage builds optimize production container images by reducing binary footprints and eliminating compiler dependencies.",
    ]

    healthcare = [
        "Electronic Health Records (EHR) systems utilize FHIR standards to enable interoperable clinical data exchange among hospitals.",
        "Deep convolutional neural networks assist radiologists in detecting early-stage pulmonary nodules and thoracic abnormalities in CT scans.",
        "Telemedicine platforms offer secure, HIPAA-compliant video consultations and remote diagnostic telemetry for chronic disease management.",
        "Genomic sequencing pipelines analyze whole-genome variants to identify targeted cancer immunotherapies and pharmacogenomic risks.",
        "Continuous glucose monitors paired with automated insulin delivery algorithms significantly improve glycemic control in diabetic patients.",
        "Clinical decision support systems integrate real-time laboratory vitals with predictive models to provide early sepsis warning alerts.",
        "Cardiovascular risk assessment algorithms combine echocardiogram imaging with biomarker analysis to forecast coronary artery disease.",
        "Wearable biosensors stream physiological vital signs to remote telemetry stations, facilitating proactive post-operative cardiac recovery.",
        "CRISPR-Cas9 gene editing therapies target monogenic genetic disorders by inserting corrective nucleotide sequences into somatic cells.",
        "Natural language processing models extract structured clinical trial inclusion criteria from unstructured oncology research notes.",
    ]

    finance = [
        "Algorithmic trading systems execute high-frequency arbitrage strategies by analyzing limit order book dynamics with sub-millisecond latency.",
        "Graph neural networks detect synthetic identity fraud and money laundering syndicates by uncovering hidden transaction topologies.",
        "Automated credit risk rating models assess borrower default probabilities using alternative cash flow data and gradient boosted trees.",
        "Portfolio optimization algorithms apply modern portfolio theory and Monte Carlo simulations to maximize risk-adjusted Sharpe ratios.",
        "Decentralized finance smart contracts enable automated liquidity pools and collateralized peer-to-peer lending on distributed ledgers.",
        "Real-time fraud prevention engines flag anomalous credit card purchases using streaming behavioral analytics and vector similarity.",
        "Central bank digital currencies require cryptographic settlement guarantees, high throughput consensus, and strict regulatory compliance.",
        "Quantitative hedge funds employ natural language processing on corporate earnings call transcripts to extract alpha sentiment signals.",
        "Regulatory reporting engines automate compliance verification for Dodd-Frank, Basel III, and anti-money laundering (AML) directives.",
        "Asset liability management frameworks simulate interest rate shocks and liquidity stress scenarios to protect commercial bank solvency.",
    ]

    ecommerce = [
        "Personalized recommendation systems use two-tower neural collaborative filtering to surface relevant items tailored to customer tastes.",
        "Dynamic pricing engines adjust product catalog prices in real-time based on competitor rates, demand elasticity, and inventory stock.",
        "Automated fulfillment warehouses deploy autonomous mobile robots to accelerate item picking, packing, and cross-dock logistics.",
        "Customer churn prediction models analyze session telemetry and shopping cart abandonment patterns to trigger retention incentives.",
        "Visual search capabilities allow mobile shoppers to upload photos of garments and discover visually similar catalog items instantly.",
        "Omnichannel supply chain orchestration balances inventory levels across national distribution hubs and local brick-and-mortar stores.",
        "Conversational AI shopping assistants resolve customer support inquiries, track order shipments, and process seamless product returns.",
        "Marketplace seller analytics dashboards provide insights into search conversion rates, return on ad spend, and buy-box ownership.",
        "Last-mile delivery route optimization algorithms minimize parcel transit times and fuel consumption using real-time traffic maps.",
        "A/B testing experimentation platforms quantify customer conversion lift across checkout funnel variants with statistical rigor.",
    ]

    categories = [
        ("Technology", technologies),
        ("Healthcare", healthcare),
        ("Finance", finance),
        ("E-commerce", ecommerce),
    ]

    documents = []
    doc_id = 1

    # Template variations to build varied, realistic documents
    templates = [
        "In the domain of {category}, {base} Furthermore, key operational benchmarks focus on {spec}.",
        "{base} Industry practitioners emphasize that this approach scales effectively under high operational throughput.",
        "Recent breakthroughs in {category} demonstrate that: {base} This capability provides significant competitive advantages.",
        "Architectural assessment: {base} Crucially, system reliability and rigorous validation remain paramount.",
        "Case study in {category}: {base} Longitudinal evaluations confirm measurable performance improvements.",
        "{base} Integrating this capability into production workflows accelerates time-to-value for enterprise teams."
    ]

    specifications = [
        "strict latency budgets under 50 milliseconds",
        "fault-tolerant distributed failover mechanisms",
        "end-to-end data encryption and regulatory compliance",
        "cost-optimized compute resource utilization",
        "automated telemetry logging and distributed tracing",
        "scalable horizontal partitioning across nodes"
    ]

    while len(documents) < count:
        cat_name, base_texts = categories[(doc_id - 1) % len(categories)]
        base = base_texts[(doc_id - 1) % len(base_texts)]
        template = templates[(doc_id - 1) % len(templates)]
        spec = specifications[(doc_id - 1) % len(specifications)]

        text = template.format(category=cat_name, base=base, spec=spec)
        documents.append({
            "id": f"doc_{doc_id:04d}",
            "text": text,
            "category": cat_name
        })
        doc_id += 1

    return documents


def run_pipeline(
    model_name: str = DEFAULT_MODEL_NAME,
    model_cache_dir: str = DEFAULT_MODEL_CACHE_DIR,
    data_path: str = DEFAULT_DATA_PATH,
    faiss_index_path: str = DEFAULT_FAISS_INDEX_PATH,
    embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
    num_docs: int = 1200,
    batch_size: int = 64
):
    """
    Execute embedding generation and FAISS indexing pipeline.
    """
    # Ensure target directories exist and resolve canonical paths
    resolved_data_path = resolve_path(data_path)
    resolved_cache_dir = resolve_path(model_cache_dir)
    resolved_index_path = resolve_path(faiss_index_path)
    resolved_embeddings_path = resolve_path(embeddings_path)

    resolved_data_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_index_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STARTING EMBEDDING GENERATION AND FAISS INDEXING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Model Name:         {model_name}")
    logger.info(f"Model Cache Dir:    {resolved_cache_dir}")
    logger.info(f"Documents Path:     {resolved_data_path}")
    logger.info(f"FAISS Index Path:   {resolved_index_path}")
    logger.info(f"Embeddings Path:    {resolved_embeddings_path}")
    logger.info(f"Document Count:     {num_docs}")

    # 1. Load or Generate Documents
    if resolved_data_path.exists():
        logger.info(f"Reading existing documents from {resolved_data_path}...")
        with open(resolved_data_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
        if len(documents) < num_docs:
            logger.info(f"Existing dataset has {len(documents)} docs. Regenerating {num_docs} documents...")
            documents = generate_synthetic_documents(count=num_docs)
    else:
        documents = generate_synthetic_documents(count=num_docs)

    # Save documents.json
    logger.info(f"Saving {len(documents)} documents to {resolved_data_path}...")
    with open(resolved_data_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    # 2. Load SentenceTransformer Model
    logger.info(f"Loading SentenceTransformer model '{model_name}'...")
    # If cached model directory already contains model files, load from there
    if (resolved_cache_dir / "modules.json").exists() or (resolved_cache_dir / "config.json").exists():
        logger.info(f"Loading cached model directly from {resolved_cache_dir}...")
        model = SentenceTransformer(str(resolved_cache_dir))
    else:
        logger.info(f"Downloading model '{model_name}' and caching to {resolved_cache_dir}...")
        model = SentenceTransformer(model_name)
        model.save(str(resolved_cache_dir))
        logger.info(f"Model successfully cached at {resolved_cache_dir}")

    # 3. Generate Embeddings
    text_corpus = [doc["text"] for doc in documents]
    logger.info(f"Encoding {len(text_corpus)} texts with batch size {batch_size}...")
    embeddings = model.encode(
        text_corpus,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    ).astype("float32")

    dimension = embeddings.shape[1]
    logger.info(f"Generated embeddings shape: {embeddings.shape} (Dimension: {dimension})")

    # Save raw embeddings
    logger.info(f"Saving raw embeddings to {resolved_embeddings_path}...")
    np.save(str(resolved_embeddings_path), embeddings)

    # 4. Initialize and Populate FAISS Index
    logger.info(f"Initializing FAISS IndexFlatL2 with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    logger.info(f"Total vectors added to FAISS index: {index.ntotal}")

    # Save FAISS index
    logger.info(f"Writing FAISS index to {resolved_index_path}...")
    faiss.write_index(index, str(resolved_index_path))

    # 5. Sanity Check Verification Search
    logger.info("-" * 60)
    logger.info("RUNNING SANITY CHECK SEARCH TEST")
    test_query = "cloud container orchestration with kubernetes"
    query_vector = model.encode([test_query], convert_to_numpy=True).astype("float32")
    k = 3
    distances, indices = index.search(query_vector, k)

    logger.info(f"Query: '{test_query}'")
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        matched_doc = documents[idx]
        logger.info(f" Rank {rank + 1}: [ID: {matched_doc['id']}] Score (L2 Distance): {dist:.4f}")
        logger.info(f"         Snippet: {matched_doc['text'][:120]}...")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings and FAISS index.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name")
    parser.add_argument("--model-cache-dir", type=str, default=DEFAULT_MODEL_CACHE_DIR, help="Local model cache path")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH, help="Output path for documents.json")
    parser.add_argument("--faiss-index-path", type=str, default=DEFAULT_FAISS_INDEX_PATH, help="Output path for faiss_index.bin")
    parser.add_argument("--embeddings-path", type=str, default=DEFAULT_EMBEDDINGS_PATH, help="Output path for embeddings.npy")
    parser.add_argument("--num-docs", type=int, default=1200, help="Number of documents to generate")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")

    args = parser.parse_args()

    run_pipeline(
        model_name=args.model_name,
        model_cache_dir=args.model_cache_dir,
        data_path=args.data_path,
        faiss_index_path=args.faiss_index_path,
        embeddings_path=args.embeddings_path,
        num_docs=args.num_docs,
        batch_size=args.batch_size
    )
