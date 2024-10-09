# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import boto3
from flask import Flask, request

# Let's use Amazon SNS
sns_client = boto3.client("sns")

app = Flask(__name__)


@app.route("/server_request")
def server_request():
    print(request.args.get("param"))
    topic_arn = "arn:aws:sns:us-east-1:445567081046:test_topic"
    sns_client.publish(TopicArn=topic_arn, Message="Hello World!")
    return "served"


if __name__ == "__main__":
    app.run(port=8082)
