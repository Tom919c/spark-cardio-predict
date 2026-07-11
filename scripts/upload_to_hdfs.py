import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_STAGING_DIR = os.path.join(BASE_DIR, "data", "staging")
DEFAULT_LOCAL_FILE = os.path.join(LOCAL_STAGING_DIR, "CVD_Standard_DWD.csv")

HDFS_URL = "http://192.168.174.128:9870"
HDFS_USER = "zhao"
HDFS_DIR = "/input/cardio_project/staging"


def upload_to_hdfs(local_path=DEFAULT_LOCAL_FILE, hdfs_dir=HDFS_DIR):
    from hdfs import InsecureClient

    filename = os.path.basename(local_path)
    hdfs_file = f"{hdfs_dir.rstrip('/')}/{filename}"

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"local file not found: {local_path}")

    client = InsecureClient(url=HDFS_URL, user=HDFS_USER)
    client.makedirs(hdfs_dir)

    if client.status(hdfs_path=hdfs_file, strict=False) is not None:
        print("hdfs file exists, deleting old file")
        client.delete(hdfs_file)
    else:
        print("hdfs file does not exist")

    client.upload(hdfs_path=hdfs_file, local_path=local_path)

    if client.status(hdfs_path=hdfs_file, strict=False) is None:
        raise RuntimeError(f"hdfs upload failed: {hdfs_file}")

    print(f"hdfs upload success: {hdfs_file}")
    return hdfs_file


if __name__ == "__main__":
    upload_to_hdfs()
