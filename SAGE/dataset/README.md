# Dataset

This directory should contain the HomeBench dataset files:

- `test_data.jsonl`: Test cases
- `home_status_method.jsonl`: Home environment configurations

## Format

### test_data.jsonl
Each line is a JSON object with:
- `id`: Test case ID
- `home_id`: Home environment ID
- `input`: User query
- `output`: Ground truth commands
- `type`: Test case type

### home_status_method.jsonl
Each line is a JSON object with:
- `home_id`: Home environment ID
- `home_status`: Device states
- `method`: Available operations

## Note

Due to size limitations, the full dataset is not included in this repository.
Please refer to the original HomeBench paper for dataset access.
