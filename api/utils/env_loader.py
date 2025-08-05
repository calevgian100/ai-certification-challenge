"""
Environment Variable Loader Utility
Provides a standalone function to load environment variables from env.yaml
"""

import os


def load_env_vars():
    """Load environment variables from env.yaml file"""
    env_vars = {}
    try:
        # Get the path to env.yaml relative to the project root
        current_file = os.path.abspath(__file__)
        # Go up from: api/utils/env_loader.py -> api -> project_root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        env_file_path = os.path.join(project_root, 'env.yaml')
        
        with open(env_file_path, 'r') as file:
            content = file.read()
            # Simple parsing for each key in the YAML file
            for line in content.splitlines():
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().strip('\'"')
                    if value:  # Only add non-empty values
                        env_vars[key] = value
                        # Set as environment variable in both original case and uppercase
                        os.environ[key] = value  # Original case (e.g., langsmith_api_key)
                        os.environ[key.upper()] = value  # Uppercase (e.g., LANGSMITH_API_KEY)
        
        print(f"✅ Loaded {len(env_vars)} environment variables from env.yaml")
        return env_vars
    except Exception as e:
        print(f"⚠️ Error loading environment variables: {e}")
        return {}
