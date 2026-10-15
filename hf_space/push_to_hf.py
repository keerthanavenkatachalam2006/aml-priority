"""
Push the AML Priority Space to Hugging Face.

Usage:
  python hf_space/push_to_hf.py --username YOUR_HF_USERNAME --space-name aml-priority

This will:
  1. Log in to Hugging Face (prompts for token or uses HF_TOKEN env var)
  2. Create the Space if it doesn't exist
  3. Upload all files from hf_space/ to the Space repo

Prerequisites:
  pip install huggingface_hub
  python hf_space/setup_space.py   # copies model artifacts into hf_space/
"""
import argparse, os, sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username",   required=True, help="Your HF username")
    parser.add_argument("--space-name", default="aml-priority", help="Space repo name")
    parser.add_argument("--token",      default=None,  help="HF token (or set HF_TOKEN env)")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, login, create_repo
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Logging in to Hugging Face (you'll be prompted for your token)...")
        login()
    else:
        login(token=token)

    api     = HfApi()
    repo_id = f"{args.username}/{args.space_name}"
    print(f"\nRepo: {repo_id}")

    # Create Space if needed
    try:
        create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio",
                    exist_ok=True, private=False)
        print(f"Space created/verified: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Upload all files from hf_space/
    space_dir = Path(__file__).parent
    exclude   = {"push_to_hf.py", "test_space.py", "__pycache__", ".DS_Store"}
    uploaded  = 0

    for f in sorted(space_dir.rglob("*")):
        if not f.is_file():
            continue
        if any(ex in str(f) for ex in exclude):
            continue
        path_in_repo = f.relative_to(space_dir).as_posix()
        print(f"  Uploading {path_in_repo} ({f.stat().st_size//1024} KB)...")
        api.upload_file(path_or_fileobj=str(f), path_in_repo=path_in_repo,
                        repo_id=repo_id, repo_type="space")
        uploaded += 1

    print(f"\nUploaded {uploaded} files.")
    print(f"Space URL: https://huggingface.co/spaces/{repo_id}")
    print("Build logs: click 'Logs' tab in your Space")


if __name__ == "__main__":
    main()
