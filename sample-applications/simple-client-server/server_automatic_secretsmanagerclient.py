# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import boto3
from flask import Flask, request

# Let's use Amazon SNS
secretsmanager_client = boto3.client("secretsmanager")

app = Flask(__name__)


@app.route("/server_request")
def server_request():
    print(request.args.get("param"))
    secret_arn = (
        "arn:aws:secretsmanager:us-east-1:445567081046:secret:yiyuanh-test-secretname-nIX0Xt"
    )
    secretsmanager_client.get_secret_value(SecretId=secret_arn)
    return "served"


if __name__ == "__main__":
    app.run(port=8082)

