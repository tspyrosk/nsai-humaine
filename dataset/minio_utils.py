import requests
import os

def minio_auth(user, password):
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

TOKEN = minio_auth(os.getenv("MINIO_USER"), os.getenv("MINIO_PASS"))