import os
import shutil

# Root HF directory
HF_HOME = "/fast/zhangy/hf_cache"
HUB_CACHE = os.path.join(HF_HOME, "hub")

os.environ["HF_HOME"] = HF_HOME
os.environ["SOFT_FILELOCK"] = "1"

from filelock import SoftFileLock
import filelock
filelock.FileLock = SoftFileLock

from huggingface_hub import snapshot_download


models = [
#    "unsloth/Qwen2.5-Coder-32B-Instruct",
    "unsloth/Qwen2.5-32B-Instruct",
#    "unsloth/Qwen2.5-Coder-32B",
    "unsloth/Qwen2.5-32B",
]


def remove_all_locks():
    for root, dirs, files in os.walk(HF_HOME):
        for f in files:
            if f.endswith(".lock"):
                path = os.path.join(root, f)
                print("Removing lock:", path)
                os.remove(path)


def repo_cache_path(repo_id):
    return os.path.join(
        HUB_CACHE,
        "models--" + repo_id.replace("/", "--")
    )


def clear_model_cache(repo_id):
    path = repo_cache_path(repo_id)

    if os.path.exists(path):
        print("Removing cached model:", repo_id)
        shutil.rmtree(path)
    else:
        print("No cache found for", repo_id)


def download_model(repo_id):
    print(f"\nDownloading {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        # cache_dir=HUB_CACHE,
        resume_download=True,
        max_workers=4
    )

    print("Finished:", repo_id)


def main():
    os.makedirs(HUB_CACHE, exist_ok=True)

    print("\nCleaning locks...")
    remove_all_locks()

    print("\nRemoving cached models...")
    for m in models:
        clear_model_cache(m)

    print("\nStarting downloads...")
    for m in models:
        download_model(m)

    print("\nAll downloads finished.")


if __name__ == "__main__":
    main()
