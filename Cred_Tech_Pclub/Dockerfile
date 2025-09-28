# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt.

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download SpaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application's code into the container
COPY..

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variables (replace with your actual keys or use build-args)
ENV API_NINJA_KEY="YOUR_API_KEY_HERE"
ENV NEWS_API_KEY="YOUR_NEWS_API_KEY"

# Command to run the Streamlit app
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]