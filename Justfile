# Run code formatting
fmt:
    uvx ruff format

# Check code formatting
check:
    uvx ruff check

# Process the raw data into a dataset suitable for training
process-data:
    uv run -m supernova.dataset

# Run weights and biases sweep
sweep:
    uv run -m supernova.sweep

# Run all tests
test:
    uv run pytest