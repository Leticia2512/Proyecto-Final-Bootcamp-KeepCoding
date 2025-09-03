FROM python:3.11.0 
  
# Set the working directory in the container  
WORKDIR /app_streamlit  
  
# Install pip requirements  
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  
  
# Copy the content of the local src directory to the working directory  
COPY . .  
  
# Specify the port number the container should expose  
EXPOSE 8501  
  
# Run the application  
CMD ["streamlit", "run", "app.py", "--server.port=8501", "server.address=0.0.0.0"] 
