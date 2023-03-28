import json
from compiled_functions import get_matching_output

print('Loading function')

directory = '/root/plagiarism-detector/script/'
s3_bucket = 'nus-sambaash'
s3_folderpath = 'plagiarism-detector/'
sentbert_model_name = 'trained_bert_model.tar.gz'
final_model_name = 'final_model.tar.gz'
ngrams_lst = [1,4,5]

def lambda_handler(event, context):
    input_doc = event['input_doc']
    input_doc_name = event['input_doc_name']
    source_docs = event['source_docs']
    
    response = get_matching_output(directory, s3_bucket, s3_folderpath, sentbert_model_name, final_model_name, ngrams_lst, source_docs, input_doc, input_doc_name)

    response_object = {}
    response_object['statusCode'] = 200
    response_object['body'] = json.dumps(response)
    
    return response_object
