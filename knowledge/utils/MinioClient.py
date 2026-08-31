from minio import Minio

minio_client=Minio(
    endpoint="218.244.155.46:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)