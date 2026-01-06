# Run code formatting
fmt:
    uvx ruff format

# Check code formatting
check:
    uvx ruff check

# Process the raw data into a dataset suitable for training
process-data:
    uv run -m supernova.preprocessing

# Run weights and biases sweep
sweep:
    uv run -m supernova.sweep

# Run all tests
test:
    uv run pytest

# Evaluate the model
eval CHECKPOINT *ARGS:
    uv run -m supernova.modeling.eval {{ CHECKPOINT }} {{ ARGS }}

# Build PDF report from markdown
build_report:
    pandoc -o reports/report.pdf reports/report.md -V geometry:margin=0.5in -V lang=polish -f markdown+raw_tex -V graphics=true
