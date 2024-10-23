# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

# Clone the specific branch of the repository
# Replace 'your-repo-url' with your actual repository URL
# and 'your-branch-name' with the branch you want to clone
RUN git clone -b your-branch-name https://github.com/your-repo-url.git .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run main.py when the container launches
CMD ["python", "main.py"]
