import requests
import os


class MinioError(RuntimeError):
    """Base class for MinIO operations that could not complete."""

    def user_message(self):
        """Message suitable for showing directly to the user."""
        return str(self)


class MinioAuthError(MinioError):
    """No usable token: credentials are missing or authentication failed."""


class MinioDownloadError(MinioError):
    """The download request reached the API but did not return the object."""

    def __init__(self, bucket_name, object_name, status_code, response_text):
        self.bucket_name = bucket_name
        self.object_name = object_name
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(
            f"MinIO download failed for {bucket_name}/{object_name}: "
            f"{status_code} {response_text}"
        )

    def user_message(self):
        location = f"`{self.bucket_name}/{self.object_name}`"
        if self.status_code in (401, 403):
            return (
                f"MinIO rejected the request for {location} ({self.status_code}). "
                "The session token is missing or has expired - re-upload the .env file "
                "in the sidebar, or restart the app to authenticate again."
            )
        if self.status_code == 404:
            return (
                f"MinIO has no object at {location} (404). Check the bucket name and the "
                "object path - the path is `bucket/path/to/file.csv`, with no leading slash."
            )
        if self.status_code >= 500:
            return (
                f"The MinIO API returned a server error ({self.status_code}) for {location}. "
                "This is on the MinIO side - retry in a moment."
            )
        return (
            f"MinIO returned {self.status_code} for {location}: {self.response_text}"
        )


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
    if not token:
        raise MinioAuthError(
            "Not authenticated with MinIO. Upload a .env file containing MINIO_USER and "
            "MINIO_PASS in the sidebar, then try again."
        )

    data_url = f"https://humaine-minio-api.euprojects.net/main_ops/download/{bucket_name}/{object_name}"
    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "application/json"
    }

    try:
        data_response = requests.get(data_url, headers=headers)
    except requests.exceptions.RequestException as e:
        raise MinioError(
            "Could not reach the MinIO API at humaine-minio-api.euprojects.net. "
            f"Check network access from this environment: {e}"
        ) from e
    print("Response", data_response, flush=True)

    if data_response.status_code != 200:
        print("Failed to download file:", data_response.status_code, data_response.text, flush=True)
        raise MinioDownloadError(
            bucket_name, object_name, data_response.status_code, data_response.text
        )

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(data_response.content)
    print("File saved", flush=True)

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