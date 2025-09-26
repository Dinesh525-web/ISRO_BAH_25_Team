# MOSDAC AI Knowledge Navigator

An intelligent AI-powered chatbot for querying and retrieving information from the MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre) portal using advanced RAG (Retrieval-Augmented Generation) technology.

## 🚀 Features

- **Intelligent Web Scraping**: Automated scraping of MOSDAC portal for real-time data
- **RAG-Powered Responses**: Context-aware responses using retrieved knowledge
- **Knowledge Graph Integration**: Semantic relationships between satellite data entities
- **Multi-Modal Interface**: Both Streamlit and Gradio interfaces available
- **Production-Ready**: FastAPI backend with monitoring and logging
- **Scalable Architecture**: Containerized deployment with Docker and Kubernetes

## 🏗️ Architecture

```
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Frontend      │ │   Backend       │ │   Data Layer    │
│  (Streamlit/    │◄┤   (FastAPI)     │◄┤  (PostgreSQL/   │
│   Gradio)       │ │                 │ │   Neo4j/Redis)  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   AI/ML Layer   │
                       │  (LangChain/    │
                       │   OpenAI/FAISS) │
                       └─────────────────┘
```
![WhatsApp Image 2025-07-09 at 19 54 42](https://github.com/user-attachments/assets/fac458bf-ea5c-46c1-9e06-a6446c52abe7)
<img width="1024" height="1536" alt="process flow diagram" src="https://github.com/user-attachments/assets/db1cbea6-ebd5-4adb-a364-056882ccff5d" />
<img width="1024" height="1536" alt="Arch_Diagram" src="https://github.com/user-attachments/assets/ae2dcbfb-fdff-49ed-a83a-a3874bd6bd09" />
<img width="1024" height="1024" alt="TeamPic" src="https://github.com/user-attachments/assets/abf0b004-e11d-4fa8-ac0c-6922638a934d" />

## 📦 Installation

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- Node.js (for frontend development)
- PostgreSQL
- Redis

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mosdac-ai-knowledge-navigator
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start services with Docker**
   ```bash
   docker-compose up -d
   ```

5. **Run the application**
   ```bash
   # Backend
   python src/backend/main.py

   # Frontend
   streamlit run src/frontend/streamlit_app/main.py
   ```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `DATABASE_URL`: PostgreSQL connection string
- `NEO4J_URI`: Neo4j database URI
- `MOSDAC_USERNAME`: MOSDAC portal username
- `MOSDAC_PASSWORD`: MOSDAC portal password

### Logging Configuration
The application uses structured logging with different levels:
- `DEBUG`: Detailed information for debugging
- `INFO`: General information about application flow
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical errors

## 🚀 Usage

### Web Interface
1. Open your browser and navigate to `http://localhost:8501` (Streamlit)
2. Enter your query about satellite data or meteorological information
3. The system will retrieve relevant information and provide contextual responses

### API Interface
```bash
curl -X POST "http://localhost:8000/api/v1/chat"      -H "Content-Type: application/json"      -d '{"query": "What are the latest INSAT-3D products?"}'
```

### Programmatic Usage
```python
from src.backend.services.rag_service import RAGService

rag_service = RAGService()
response = rag_service.query("Tell me about MOSDAC data products")
print(response)
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run all tests with coverage
pytest --cov=src tests/
```

## 📊 Monitoring

The application includes comprehensive monitoring:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Structured Logging**: JSON-formatted logs
- **Health Checks**: Application health monitoring

Access monitoring dashboards:
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Scale services
docker-compose up --scale backend=3
```

## ☸️ Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f deployments/kubernetes/

# Check deployment status
kubectl get pods -n mosdac-ai
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ISRO MOSDAC team for providing satellite data access
- OpenAI for GPT models
- LangChain community for RAG framework
- Streamlit team for the web framework

## 📞 Support

For support, please open an issue on GitHub or contact the development team.
