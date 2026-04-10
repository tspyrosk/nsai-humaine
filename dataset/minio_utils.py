import requests
import os

def minio_auth(user, password):
    # Skip authentication if credentials are not provided
    if not user or not password:
        return None

    auth_url = "https://humaine-minio-api.euprojects.net/auth/auth"
    credentials = {
        "username": user,
        "password": password
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    auth_response = requests.post(auth_url, data=credentials, headers=headers)

    if auth_response.status_code == 200:
        json_data = auth_response.json()
        token = json_data["access_token"]
        print("Received token.")
        return token
    else:
        print("Authentication failed:", auth_response.status_code, auth_response.text)
        return None

def minio_download(token, bucket_name, object_name, file_path):
    data_url = f"https://humaine-minio-api.euprojects.net/main_ops/download/{bucket_name}/{object_name}"
    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "application/json"
    }

    data_response = requests.get(data_url, headers=headers)
    print("Response", data_response, flush=True)

    if data_response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(data_response.content)
        print("File saved", flush=True)
    else:
        print("Failed to download file:", data_response.status_code, data_response.text, flush=True)

def minio_upload(token, bucket_name, object_name, file_path):
    upload_url = "https://humaine-minio-api.euprojects.net/main_ops/upload"
    with open(file_path, "rb") as f:
        files = {
            "file": (object_name, f, "text/json")
        }
        data = {
            "bucket_name": bucket_name,
            "object_name": object_name
        }
        headers = {
            "Authorization": f"Bearer {token}"
        }

        response = requests.post(upload_url, data=data, files=files, headers=headers)

        if response.status_code == 200:
            print("Upload successful:", response.text, flush=True)
        else:
            print("Upload failed:", response.status_code, response.text, flush=True)

def minio_read_json(token, bucket_name, object_name):
    """Download a JSON object from MinIO and return it as a dict. Returns None if not found."""
    data_url = f"https://humaine-minio-api.euprojects.net/main_ops/download/{bucket_name}/{object_name}"
    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "application/json"
    }
    response = requests.get(data_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

def minio_write_json(token, bucket_name, object_name, data):
    """Serialize data as JSON and upload it to MinIO."""
    import json
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, tmp)
    tmp.close()
    minio_upload(token, bucket_name, object_name, tmp.name)
    os.unlink(tmp.name)

# Token is now managed via st.session_state in main.py to avoid repeated auth on Streamlit reruns