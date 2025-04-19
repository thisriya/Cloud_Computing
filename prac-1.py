import boto3
import json

from botocore.exceptions import ClientError

client = boto3.client("bedrock-runtime", region_name="us-west-2")

model_id = "meta.llama3-8b-instruct-v1:0"

prompt = "Best places to visit in banaras"

formatted_prompt = f"""
<|begin_f_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

native_request = {
    "prompt": formatted_prompt,
    "max_gen_len": 512,
    "temperature": 0.5,
    }

request = json.dumps(native_request)

try:
    response = client.invoke_model(modelId=model_id, body=request)

except(ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

model_response = json.loads(response["body"].read())

response_text = model_response["generation"]
print(response_text)