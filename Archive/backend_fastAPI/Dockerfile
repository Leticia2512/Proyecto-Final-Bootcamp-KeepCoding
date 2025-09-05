FROM python:3.11.0 
  
# Set the working directory in the container  
WORKDIR /app  
  
# Install pip requirements  
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  
  
# Copy the content of the local src directory to the working directory  
COPY . .  
  
# Specify the port number the container should expose  
EXPOSE 8080  
  
# Run the application  
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 
