# Vector Database with Qdrant - ArXiv Papers Semantic Search

A comprehensive implementation of semantic search for ArXiv research papers using Qdrant vector database, featuring multiple optimization stages from basic data loading to advanced performance tuning with quantization.

## 🎯 Project Overview

This project demonstrates a complete workflow for building and optimizing a semantic search system for academic papers using:
- **Qdrant** as the vector database
- **OpenAI text-embedding-ada-002** for query embeddings
- **ArXiv dataset** with pre-computed embeddings
- **Performance optimization** techniques including HNSW parameters and quantization

The project is structured as progressive stages, each building upon the previous to showcase different aspects of vector search implementation and optimization.

## 📋 Requirements

### Dependencies
```
fastapi[all]
openai
pydantic
python-dotenv
qdrant_client
requests
colorama
matplotlib
```

### External Requirements
- **Docker** - For running Qdrant locally
- **OpenAI API Key** - For generating query embeddings
- **ArXiv Dataset** - ML papers with embeddings (1536-dimensional vectors)

## 🚀 Installation & Setup

### 1. Docker Setup for Qdrant
```bash
# Start Qdrant using Docker
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_LITELLM_API_KEY=your_openai_api_key_here
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Data Preparation
Ensure you have the ArXiv dataset with embeddings at:
```
C:\Shashank_work\arxiv_dataset\ml-arxiv-embeddings.json
```

## 📁 Project Structure

### Core Implementation Stages

#### **Stage 1: Data Loading** (`stage1_semantic_search.py`)
- Establishes Qdrant connection (localhost:6333)
- Creates collection with 1536-dimensional vectors using cosine distance
- Implements batch loading of ArXiv papers with embeddings
- Features progress tracking with colored output

#### **Stage 2: Basic Similarity Search** (`stage2_semantic_search.py`)
- Implements paper-to-paper similarity search
- Uses existing paper ID to find similar papers
- Demonstrates scroll filtering and vector querying
- Returns top-K most similar paper IDs

#### **Stage 3: Natural Language Queries** (`stage3_semantic_search_with_openai_query_embedding.py`)
- Integrates OpenAI text-embedding-ada-002 model
- Converts natural language queries to embeddings
- Enables semantic search with text input
- Uses custom LiteLLM endpoint for embedding generation

#### **Stage 4: Advanced Filtering** 
- `stage4_semantic_search_with_openai_query_embedding_author_name_filter.py`: Author-based filtering
- `stage4_plus_fastapi_result_retrieval.py`: FastAPI integration for web API

#### **Stage 6: HNSW Performance Analysis** (`stage6_precision_calculation_hnsw_search_qdrant.py`)
- Analyzes HNSW (Hierarchical Navigable Small World) algorithm performance
- Compares ANN (Approximate Nearest Neighbor) vs exact search
- Measures precision and query time trade-offs

#### **Stage 7: EF Parameter Optimization** (`stage7_semantic_search_efhnsw.py`)
- Tests different `ef` (exploration factor) values
- Optimizes search quality vs speed balance
- Generates performance metrics for various configurations

#### **Stage 8: Advanced Optimization** 
- `stage8_optimised_version.py`: Single optimized implementation
- `stage8_optimised_version_many_runs.py`: Multiple run analysis
- `stage8_semantic_search_m_ef_construct.py`: M and EF construct parameter testing

#### **Stage 9: Quantization Performance** (`stage9_semantic_search_quantization.py`)
- Implements INT8 scalar quantization
- Compares rescoring vs non-rescoring approaches
- Generates detailed performance analysis with visualizations
- **Key Results**: Rescoring achieves 99.9% precision (0.0364s) vs 84.8% precision (0.0202s) without rescoring

### Performance Results & Analysis

#### **Quantization Results** (`quantization_results_with_and_without_rescoring.json`)
```json
[
    {
        "rescore": true,
        "oversampling": 2,
        "precision": 0.999,
        "ann_time": 0.0364
    },
    {
        "rescore": false,
        "oversampling": 2,
        "precision": 0.848,
        "ann_time": 0.0202
    }
]
```

#### **Visual Analysis**
- `quantization_results_rescore_split_plot.png`: Performance comparison charts
- `m_and_ef_construct_test_results2.png`: HNSW parameter optimization results

### Additional Components

#### **Test Files**
- `tests.py`: Main test suite
- `test/tests.py`: Additional test cases
- `test_queries_embeddings.json`: Test dataset for performance evaluation

#### **Additional Files**
- `alternate_solution_file.py`: Alternative implementation approaches
- `embeddings_from_openai.py`: OpenAI embedding utilities
- `sparse_vectors.py`: Sparse vector handling
- `verify_collections_through_http_request.py`: HTTP API verification

## 🔧 Usage Examples

### Basic Semantic Search
```python
from stage3_semantic_search_with_openai_query_embedding import top_5_similar_papers

# Search with natural language query
query = "attention mechanism in deep learning"
results = top_5_similar_papers(query)
print(f"Similar papers: {results}")
```

### Performance Testing
```python
from stage9_semantic_search_quantization import get_search_results_precision_and_times

# Test quantization performance
rescore_config = {"rescore": True, "oversampling": 2}
precision, avg_time = get_search_results_precision_and_times(test_queries, rescore_config)
```

## 📊 Key Performance Insights

### Optimization Trade-offs
1. **Rescoring**: Higher precision (99.9%) but slower (36.4ms)
2. **No Rescoring**: Lower precision (84.8%) but faster (20.2ms)
3. **Quantization**: Reduces memory usage with configurable precision trade-offs

### HNSW Parameters
- **M (max connections)**: Affects index build time and search quality
- **EF (exploration factor)**: Balances search speed vs accuracy
- **EF_CONSTRUCT**: Impacts index building performance

## 🛠️ Development Workflow

1. **Data Loading**: Use Stage 1 to populate Qdrant with ArXiv embeddings
2. **Basic Testing**: Validate similarity search with Stage 2
3. **Query Integration**: Enable natural language queries with Stage 3
4. **Performance Tuning**: Use Stages 6-9 for optimization analysis
5. **Production Setup**: Implement FastAPI integration from Stage 4

## 🚦 Docker & Infrastructure

The project includes comprehensive Docker setup scripts in the `Readme/` directory:
- `start_qdrant_docker.bat`: Launch Qdrant container
- `setup_qdrant_ui.bat`: Setup Qdrant web interface
- `stop_all_docker_containers.bat`: Cleanup Docker resources

## 📈 Performance Metrics

- **Vector Dimension**: 1536 (text-embedding-ada-002)
- **Distance Metric**: Cosine similarity
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Quantization**: INT8 scalar with 0.99 quantile
- **Search Precision**: Up to 99.9% with rescoring enabled

## 🔍 Research Applications

This implementation is ideal for:
- **Academic Research**: Finding related papers by topic
- **Literature Reviews**: Discovering relevant publications
- **Knowledge Discovery**: Exploring research connections
- **Performance Benchmarking**: Vector database optimization studies

## 📚 Technical Stack

- **Vector Database**: Qdrant (Docker deployment)
- **Embeddings**: OpenAI text-embedding-ada-002
- **Backend**: Python with FastAPI
- **Visualization**: Matplotlib for performance charts
- **Data Format**: JSONL for ArXiv dataset
- **Optimization**: HNSW indexing with quantization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Project Status**: Production-ready with comprehensive performance analysis and optimization features.

**Last Updated**: August 13, 2025