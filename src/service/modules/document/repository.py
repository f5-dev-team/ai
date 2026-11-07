import boto3

from src.common.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    S3_BUCKET_NAME,
)


class DocumentRepository:
    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )

    def list_files(self):
        response = self.s3.list_objects_v2(
            Bucket=S3_BUCKET_NAME
        )

        return [
            item["Key"]
            for item in response.get("Contents", [])
        ]