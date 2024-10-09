# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import boto3
from flask import Flask, request

# Let's use Amazon S3
kinesis_client = boto3.client("kinesis")

app = Flask(__name__)


@app.route("/server_request")
def server_request():
    print(request.args.get("param"))
    stream_arn = "arn:aws:kinesis:us-east-1:445567081046:stream/yiyuanh-us-east-1-test-stream"
    stream_name = "yiyuanh-us-east-1-test-stream"
    kinesis_client.describe_stream(StreamARN=stream_arn)
    return "served"


if __name__ == "__main__":
    app.run(port=8082)

