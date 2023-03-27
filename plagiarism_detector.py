import json
from utils.compiled_functions import get_matching_output

print('Loading function')

directory = '/root/plagiarism-detector/script/'
s3_bucket = 'nus-sambaash'
s3_folderpath = 'plagiarism-detector/'
sentbert_model_name = 'trained_bert_model.tar.gz'
xgboost_model_name = 'xgboost_model.tar.gz'
ngrams_lst = [1,4,5]

# source_docs = [{'source_doc_name': 'source_string_name', 
#               'source_doc': "The pride of the Russians did not suffer in consequence.  While poetry naturally precedes dramatic art, the drama, on the other hand, cannot attain any degree of excellence where the theater is in such a miserable state. It is now scarcely half a century since the effort was begun to remove the total want of scientific culture in the Russian nation, but what are fifty years for such a purpose, in so enormous a country? The number of those who have received the scientific stimulus and been carried to a degree of intellectual refinement is very small, and the happy accident by which a man of genius appears among the small number must be very rare. And in this connection it is noteworthy, that the Russian who feels himself called to artistic production almost always shows a tendency to epic composition.  The difficulties of form appear terrible to the Russian. In romance-writing the form embarrasses him less, and accordingly they almost all throw themselves into the making of novels.  As is generally the case in the beginning of every nation's literature, any writer in Russia is taken for a miracle, and regarded with stupor. The dramatist Kukolnik is an example of this. He has written a great deal for the theater, but nothing in him is to be praised so much as his zeal in imitation. It must be admitted that in this he possesses a remarkable degree of dexterity. He soon turned to the favorite sphere of romance writing, but in this also he manifests the national weakness. In every one of his countless works the most striking feature is the lack of organization. They were begun and completed without their author's ever thinking out a plot, or its mode of treatment."}]
# input_doc = "Pride Russians suffered accordingly.  While naturally precedes dramatic art poetry, drama, on the other hand, can achieve a degree of excellence where the theater is in a miserable state. It is now barely half a century from the beginning was an effort to remove all the lack of scientific culture in the Russian nation, but what are fifty years for such purpose, in a country so huge? The number of received scientific stimulus and was carried to a degree of intellectual sophistication is very low, and a happy accident that is genius among the small number must be very rare. And in this respect, it is noteworthy that the Russians who feels called to the artistic production almost always show a trend of view of epic composition.  Difficulties as the Russian seem terrible. In love, written as he embarrasses less and therefore have almost all the throws in the making of novels.  As is generally the case in the early literature of every nation, every writer in Russia is taken for a miracle, and looked with astonishment. Kukolnik playwright is an example. He wrote a good deal for the theater, but nothing is as it should be praised as much as his zeal in imitation. It must be admitted that in this he possesses a remarkable degree of dexterity. He soon became the area's favorite love, but this also he shows weakness at the national level. In each of his numerous works the most striking feature is the lack of organization. They were completed with no beginning and author ever thought a plot, or mode of treatment"
# input_doc_name = 'input_string_name'

def lambda_handler(event, context):
    input_doc = event['input_doc']
    input_doc_name = event['input_doc_name']
    source_docs = event['source_docs']
    
    response = get_matching_output(directory, s3_bucket, s3_folderpath, sentbert_model_name, xgboost_model_name, ngrams_lst, source_docs, input_doc, input_doc_name)

    response_object = {}
    response_object['statusCode'] = 200
    response_object['body'] = json.dumps(response)
    
    return response_object
