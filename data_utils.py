import json

def load_prompts(file_path):
    """Load a JSON file and return its content."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    prompts = [item["prompt"] for item in data]
    index = [item["JSON_idx"] for item in data]

    return prompts,index