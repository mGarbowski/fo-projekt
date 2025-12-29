# Run code formatting
fmt:
    uvx ruff format

# Check code formatting
check:
    uvx ruff check

# Process the raw data into a dataset suitable for training
process-data:
    uv run -m supernova.dataset

sweep:
    uv run -m supernova.sweep
