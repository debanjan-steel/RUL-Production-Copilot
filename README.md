# 🏭 Predictive Maintenance RUL Copilot (MLOps Edition)

An end-to-end, production-ready Machine Learning pipeline for predicting the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset. 

Unlike standard static notebooks, this project wraps a **CNN-Transformer model** inside an agentic Copilot and serves it via a **FastAPI** REST endpoint. It features uncertainty-aware predictions (via MC Dropout) and automatically escalates to a human operator when confidence is low.

## 🚀 Key Engineering Features
* **Advanced Architecture:** CNN-Transformer hybrid handling temporal sequences and local sensor noise.
* **Uncertainty-Aware Abstention:** Uses Monte Carlo Dropout to gauge prediction confidence; refuses to predict if uncertainty exceeds safety thresholds.
* **Production Serving:** Model wrapped in a high-performance FastAPI endpoint.
* **Containerized:** Fully Dockerized for environment-agnostic deployment.
* **CI/CD Pipeline:** Automated testing and build processes via GitHub Actions.

## 📂 Repository Structure
* `/src` - Core ML logic, model architectures, and training loops.
* `/api` - FastAPI application, endpoints, and Pydantic schemas.
* `/kb` - Markdown knowledge base for the agent's Retrieval-Augmented Generation (RAG).
* `.github/workflows` - CI/CD pipeline configurations.

## 🛠️ Quickstart

**1. Clone the repository:**
`git clone https://github.com/YourUsername/rul-production-copilot.git`
`cd rul-production-copilot`

**2. Run the application via Docker:**
`docker build -t rul-copilot .`
`docker run -p 8000:8000 rul-copilot`

**3. Test the Endpoint:**
Navigate to `http://localhost:8000/docs` to interact with the API via the Swagger UI.