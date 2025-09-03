fastapi dev backend_fastAPI/main.py --reload
uvicorn main:app --host 0.0.0.0 --port 8080
streamlit run frontend_Streamlit/app2.py

# Create docker image
docker build -t ocular_app .
# Check docker images
docker images
# Run docker container 
docker run -d -p 8080:8080 --name ocular_app_container ocular_app
# Check running container
docker ps -a
# Remove container
docker rm ocular_app_container
# Stop docker container
docker stop ocular_app_container
# Restart docker container
docker start ocular_app_container

# Create docker image fastapi
docker build -t fastapi_app .
# Run docker container fastapi
docker run -d -p 8080:8080 --name fastapi_app_container fastapi_app
# Create docker image streamlit
docker build -t streamlit_app .
# Run docker container streamlit
docker run -d -p 8501:8501 --name streamlit_app_container streamlit_app

# Run docker compose
docker compose up

# Create docker image
docker build -t gcr.io/{vuestro-proyecto}/bd-15-pruebas-tohello .
# Upload image to container registry
docker push gcr.io/{vuestro-proyecto}/bd-15-pruebas-tohello
# Deploy image
gcloud run deploy bd-15-pruebas-tohello --image gcr.io/{vuestro-proyecto}/bd-15-pruebas-tohello --platform managed --region europe-west4
