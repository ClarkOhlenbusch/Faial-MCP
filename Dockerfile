# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project's dependency file
COPY pyproject.toml .

# Install the dependencies
RUN pip install -e .

# Copy the rest of the application's source code
COPY . .

# Run the MCP server
CMD ["python", "-m", "faial_mcp_server.server"]
