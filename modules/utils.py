import json
import os
from functools import wraps


def save_progress(progress_file, data, **kwargs):
    with open(progress_file, "w") as f:
        json.dump(data, f, **kwargs)


def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return json.load(f)
    return None


def resumable(progress_file):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_idx = kwargs.pop("start_idx", 0)
            data = kwargs.get("data", args[0] if args else None)
            if data is None:
                raise ValueError("Data must be provided")

            progress_data = load_progress(progress_file)
            if progress_data is not None:
                data = progress_data
                saved_start_idx = len(data) - sum(
                    1 for item in data if "context" not in item
                )
                start_idx = max(start_idx, saved_start_idx)
                print(f"Resuming from index {start_idx}")

            result = func(
                *args, **kwargs, start_idx=start_idx, progress_file=progress_file
            )
            return result

        return wrapper

    return decorator
