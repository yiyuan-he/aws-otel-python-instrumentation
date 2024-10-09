# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import boto3
from flask import Flask, request

# Let's use Amazon SNS
stepfunctions_client = boto3.client("stepfunctions")

app = Flask(__name__)


@app.route("/server_request")
def server_request():
    print(request.args.get("param"))
    state_machine_arn = "arn:aws:states:us-east-1:445567081046:stateMachine:MyStateMachine-9q1h125em"
    activity_arn = "arn:aws:states:us-east-1:445567081046:activity:yiyuanh-us-east-1-test-activity"
    stepfunctions_client.describe_state_machine(stateMachineArn=state_machine_arn)
    stepfunctions_client.describe_activity(activityArn=activity_arn)
    return "served"


if __name__ == "__main__":
    app.run(port=8082)

