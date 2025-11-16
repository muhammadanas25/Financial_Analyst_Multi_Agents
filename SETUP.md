# Complete Setup and Execution Guide

This guide provides step-by-step instructions to set up and run the FAB Financial Analysis System from scratch.

## Prerequisites

### Required Software
- **Python 3.10+** ([Download](https://www.python.org/downloads/))
- **Docker Desktop** ([Download](https://www.docker.com/products/docker-desktop/))
- **Git** ([Download](https://git-scm.com/downloads))

### Required API Keys
- **OpenAI API Key** (for GPT-4 and embeddings)
  - Get from: https://platform.openai.com/api-keys
  - Required for LLM-based agents

### Hardware Requirements
- **Minimum**: 8GB RAM, 4 CPU cores, 10GB free disk space
- **Recommended**: 16GB RAM, 8 CPU cores, 20GB free disk space

---

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Financial_Analyst_Multi_Agents
```

### 2. Create Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

Your terminal prompt should now show `(venv)` prefix.

### 3. Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

This will install:
- LangChain & LangGraph (agent orchestration)
- Weaviate client (vector database)
- Document parsers (PyMuPDF, pdfplumber, docling)
- SentenceTransformers (embeddings)
- OpenAI client
- And all other dependencies

**Installation time**: 5-10 minutes depending on internet speed.

### 4. Set Up Environment Variables

```bash
# Copy template
cp .env.template .env

# Edit .env file
nano .env  # or use your preferred editor
```

**Required configuration in `.env`**:

```env
# Required - Get from https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional - For monitoring (can leave as is for now)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=fab-financial-analyst

# System Configuration (defaults are fine)
PRIMARY_LLM_MODEL=gpt-4-turbo-preview
FALLBACK_LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=2048
CHUNK_OVERLAP=200
HYBRID_SEARCH_ALPHA=0.3
CONFIDENCE_THRESHOLD=0.70
```

**Important**: Replace `sk-your-openai-api-key-here` with your actual OpenAI API key!

### 5. Start Weaviate Vector Database

```bash
# Start Weaviate using Docker Compose
docker-compose up -d

# Verify Weaviate is running
docker ps
```

You should see:
```
CONTAINER ID   IMAGE                              STATUS         PORTS
xxxxxx         semitechnologies/weaviate:1.24.1   Up X minutes   0.0.0.0:8080->8080/tcp
```

**Check Weaviate is healthy**:
```bash
curl http://localhost:8080/v1/.well-known/ready
```

Should return: `{"status": "healthy"}`

**Troubleshooting**:
- If port 8080 is already in use, modify `docker-compose.yml` to use a different port
- On Windows, make sure Docker Desktop is running
- Check logs: `docker-compose logs weaviate`

### 6. Create Required Directories

```bash
# These should already exist, but just in case:
mkdir -p data output logs
```

### 7. Verify Installation

```bash
# Test Python imports
python -c "import langchain, weaviate, sentence_transformers; print('✓ All imports successful')"

# Test Weaviate connection
python -c "import weaviate; client = weaviate.connect_to_local(); print('✓ Weaviate connection successful'); client.close()"
```

---

## Running the System

### Phase 1: Document Processing (One-time Setup)

Process the FAB Q1 2025 documents and extract structured information.

```bash
python examples/process_fab_documents.py
```

**What this does**:
1. Parses 3 PDF files using multiple parsers
2. Extracts metadata (company, fiscal period, etc.)
3. Creates element-based chunks
4. Saves results to `output/` directory

**Expected output**:
```
FAB Q1 2025 Document Processing
================================================================================
Processing: FAB-Earnings-Presentation-Q1-2025.pdf
--------------------------------------------------------------------------------
✓ Processed FAB-Earnings-Presentation-Q1-2025.pdf: 156 chunks

Processing: FAB-FS-Q1-2025-English.pdf
--------------------------------------------------------------------------------
✓ Processed FAB-FS-Q1-2025-English.pdf: 243 chunks

Processing: FAB-Q1-2025-Results-Call.pdf
--------------------------------------------------------------------------------
✓ Processed FAB-Q1-2025-Results-Call.pdf: 89 chunks

Overall Statistics
================================================================================
Documents Processed: 3
Total Pages: 80
Total Chunks: 488
Average Quality: 0.923
```

**Time**: 2-3 minutes

**Output location**: `output/` directory contains:
- `FAB-Earnings-Presentation-Q1-2025/metadata.json`
- `FAB-FS-Q1-2025-English/chunks_full.json`
- `ingestion_summary.json`
- Extraction quality reports

### Phase 2: Ingest to Vector Database

Load processed documents into Weaviate for semantic search.

```bash
python scripts/ingest_to_weaviate.py
```

**What this does**:
1. Reads processed chunks from Phase 1
2. Generates embeddings using SentenceTransformers
3. Stores in Weaviate with metadata
4. Tests retrieval

**Expected output**:
```
FAB Document Ingestion to Weaviate
================================================================================
Step 1: Processing Documents
--------------------------------------------------------------------------------
✓ Processed FAB-Earnings-Presentation-Q1-2025.pdf: 156 chunks
✓ Processed FAB-FS-Q1-2025-English.pdf: 243 chunks
✓ Processed FAB-Q1-2025-Results-Call.pdf: 89 chunks

Total chunks to ingest: 488

Step 2: Connecting to Weaviate
--------------------------------------------------------------------------------
✓ Connected to Weaviate

Step 3: Ingesting Chunks to Weaviate
--------------------------------------------------------------------------------
Adding 488 chunks to Weaviate...
Added batch 1/5
Added batch 2/5
...
✓ Successfully added 488 chunks to Weaviate

Ingestion Complete!
================================================================================
Total chunks ingested: 488
Total objects in database: 488

Testing Retrieval
--------------------------------------------------------------------------------
Test query: What was FAB's total revenue in Q1 2025?

Found 3 results:
1. Score: 0.842 | FAB-FS-Q1-2025-English.pdf | Page 5
   Content preview: Total operating income for Q1 2025 was AED 12.5 billion...

✓ System ready for queries!
```

**Time**: 5-10 minutes (depends on CPU for embeddings)

**Note**: The first run downloads the embedding model (~400MB). Subsequent runs are faster.

### Phase 3: Query the System

Now you can ask questions about FAB's Q1 2025 financials!

```bash
python scripts/query_system.py
```

**Choose an option**:
```
FAB Financial Analysis System
================================================================================

Options:
1. Run example queries
2. Interactive mode
3. Exit

Choice (1-3):
```

#### Option 1: Run Example Queries

Runs 4 predefined queries demonstrating different capabilities:

1. **Simple Retrieval**: "What was FAB's total revenue in Q1 2025?"
2. **Calculation**: "Calculate FAB's revenue growth from Q1 2024 to Q1 2025"
3. **Comparison**: "Compare FAB's net income in Q1 2025 vs Q1 2024"
4. **Analysis**: "What were the key drivers of FAB's performance in Q1 2025?"

**Example output**:
```
================================================================================
Query 1/4
================================================================================
Q: What was FAB's total revenue in Q1 2025?

Processing...
InputValidationAgent: Classified as retrieval (simple)
RetrievalAgent: Retrieved 10 documents (avg score: 0.782)
SynthesisAgent: Response synthesized (confidence: 0.89)

A: Based on First Abu Dhabi Bank's Q1 2025 financial statements, total revenue
was AED 12.5 billion [Document 1, Page 5], representing an increase from the
prior year quarter.

✓ Confidence: 89.0%

Sources:
1. FAB-FS-Q1-2025-English.pdf, Page 5 (Q1 2025) [Score: 0.842]
2. FAB-Earnings-Presentation-Q1-2025.pdf, Page 12 (Q1 2025) [Score: 0.798]

Reasoning Steps:
  - Retrieved 10 relevant documents with average score 0.782
  - Synthesized response with 2 citations (confidence: 0.89)

Agent Sequence: InputValidationAgent → RetrievalAgent → SynthesisAgent
```

#### Option 2: Interactive Mode

Ask your own questions!

```
Interactive Query Mode
================================================================================

Enter your questions about FAB's Q1 2025 financials.
Type 'exit' to quit, 'examples' to see example queries.

Your question: What was the net profit margin in Q1 2025?

Processing...

[Agent processing happens...]

A: FAB's net profit margin for Q1 2025 was 31.2% [Document 1, Page 8],
calculated as net profit (AED 3.9 billion) divided by total revenue
(AED 12.5 billion).

The calculation was verified: 3.9 / 12.5 = 0.312 (31.2%)

✓ Confidence: 92.0%

Sources:
1. FAB-FS-Q1-2025-English.pdf, Page 8 (Q1 2025) [Score: 0.871]

Show details? (y/n): y

Intent: calculation
Complexity: simple
Documents Retrieved: 10
Confidence: 92.0%

Reasoning Steps:
  - Retrieved 10 relevant documents with average score 0.801
  - Performed calculations: {'net_profit_margin': 0.312}
  - Synthesized response with 1 citations (confidence: 0.92)

Agent Sequence: InputValidationAgent → RetrievalAgent → CalculationAgent → SynthesisAgent
```

---

## Testing Different Query Types

### Simple Retrieval
```
What currency does FAB report in?
What is FAB's ticker symbol?
What accounting standards does FAB use?
```

### Calculations
```
Calculate the debt-to-equity ratio for Q1 2025
What is the return on equity for Q1 2025?
Calculate the operating expense ratio
```

### Comparisons
```
Compare revenue in Q1 2025 vs Q1 2024
How did net income change from Q4 2024 to Q1 2025?
Compare loan growth across quarters
```

### Analysis
```
What were the main drivers of revenue growth in Q1 2025?
Analyze FAB's asset quality in Q1 2025
What risks does FAB mention in their Q1 2025 reports?
```

---

## Stopping the System

### Stop Weaviate

```bash
# Stop Weaviate but keep data
docker-compose stop

# Stop and remove containers (data is preserved in volume)
docker-compose down

# Stop and remove containers AND data
docker-compose down -v  # ⚠️ This deletes all ingested documents!
```

### Deactivate Python Environment

```bash
deactivate
```

---

## Troubleshooting

### Issue: "Failed to connect to Weaviate"

**Solution**:
```bash
# Check if Weaviate is running
docker ps

# If not running, start it
docker-compose up -d

# Check logs
docker-compose logs weaviate

# Test connection
curl http://localhost:8080/v1/.well-known/ready
```

### Issue: "No documents in database"

**Solution**:
```bash
# Run ingestion script
python scripts/ingest_to_weaviate.py
```

### Issue: "OpenAI API key not found"

**Solution**:
```bash
# Make sure .env exists
ls -la .env

# Edit .env and add your API key
nano .env

# Verify it's set
python -c "from config.config import settings; print(settings.openai_api_key[:10])"
```

### Issue: "Module not found" errors

**Solution**:
```bash
# Make sure virtual environment is activated
which python  # Should show path with 'venv'

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: Embeddings are slow

**Solution**:
- First run downloads the model (~400MB)
- Subsequent runs use cached model
- Consider using smaller embedding model in .env:
  ```env
  EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Faster, smaller
  ```

### Issue: Out of memory during ingestion

**Solution**:
- Reduce batch size in `scripts/ingest_to_weaviate.py`:
  ```python
  vector_store.add_chunks(all_chunks, batch_size=50)  # Reduce from 100
  ```

---

## Directory Structure After Setup

```
Financial_Analyst_Multi_Agents/
├── data/                           # Input PDFs
│   ├── FAB-Earnings-Presentation-Q1-2025.pdf
│   ├── FAB-FS-Q1-2025-English.pdf
│   └── FAB-Q1-2025-Results-Call.pdf
├── output/                         # Processing results
│   ├── FAB-Earnings-Presentation-Q1-2025/
│   │   ├── metadata.json
│   │   ├── chunks.json
│   │   ├── chunks_full.json
│   │   └── extraction_report.txt
│   ├── FAB-FS-Q1-2025-English/
│   └── ingestion_summary.json
├── logs/                           # Application logs
│   ├── fab_processing_*.log
│   ├── ingestion_*.log
│   └── query_*.log
├── venv/                           # Python virtual environment
├── .env                            # Your environment variables
└── docker-compose.yml              # Weaviate configuration
```

---

## Next Steps

Once you have the system running:

1. **Explore Example Queries**: Run the predefined examples to see capabilities
2. **Ask Custom Questions**: Use interactive mode for your specific questions
3. **Review Logs**: Check `logs/` directory for detailed execution traces
4. **Inspect Results**: Look at `output/` for document processing quality
5. **Monitor Weaviate**: Visit http://localhost:8080/v1 for Weaviate console

---

## Getting Help

- **Documentation**: See `README.md` for system architecture
- **Execution Plan**: See `execution_plan.md` for comprehensive technical details
- **Logs**: Check `logs/` directory for detailed traces
- **Issues**: Review error messages and check troubleshooting section above

---

**Congratulations! Your FAB Financial Analysis System is ready to use!** 🎉
