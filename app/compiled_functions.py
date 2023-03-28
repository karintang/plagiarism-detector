import json

import nltk
import pandas as pd

nltk.download("punkt", download_dir="nltk_data/")
nltk.download("stopwords", download_dir="nltk_data/")
import difflib
import tarfile

import boto3
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import tokenize
from sentence_transformers import SentenceTransformer

######## DIRECT MATCHING FUNCTIONS ########

def get_textmatcher_library(bucket):
    """
    Download pre-build textmatcher library from S3 bucket and runs it

    Args:
        bucket (str): Name of S3 bucket.

    Returns:
        code (class): Pre-built textmatcher class.
    """
    textmatcher_key = 'plagiarism-detector/textmatcher.py'

    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=textmatcher_key)
    code = response['Body'].read().decode('utf-8')
    
    return code
    
exec(get_textmatcher_library(s3_bucket))

def read_s3_df(bucket, filepath):
    """
    Returns DataFrame of CSV file downloaded from S3 bucket.

    Args:
        bucket (str): Name of S3 bucket.
        filepath (str): Filepath of CSV file in S3.

    Returns:
        df (pd.DataFrame): Dataframe of CSV file downloaded.
    """
    data_location = 's3://{}/{}'.format(bucket, filepath)
    df = pd.read_csv(data_location, index_col=[0])
    
    return df

def get_preprocessed_sent(input_doc):
    """
    Returns input document, split by sentences

    Args:
        input_doc (str): Input document.
    
    Returns:
        res (list[tuple]): Input document, split by sentences. Contains tuple (start_ind, end_ind, sentence)
    """
    input_doc = str(input_doc)
    input_doc = input_doc.replace('\n', '')
    input_text_lst = tokenize.sent_tokenize(input_doc)
    res = []
    start_char = 1

    for s in range(len(input_text_lst)):
        res.append({'sentence': input_text_lst[s], 'start_char_index': start_char, 'end_char_index': start_char + len(input_text_lst[s])-1})
        start_char = start_char + len(input_text_lst[s])
    
    return res

def get_matching_texts(input_text_lst, source_doc):
    """
    Returns list of dictionary of matching texts 
        (input_doc_text, input_doc_start, input_doc_end, source_doc_text, 
        source_doc_start, source_doc_end, source_doc_id, cosine_similarity_score) &
        the sentence indices of input documents that are flagged as direct matches.
    
    Args:
        input_text_lst (list): Input document of interest, split by sentences.
        source_doc (string): Source input document.

    Returns:
        output_lst (list): List of dictionary of matching texts and their details.
        match_lst (list): List of dictionary of direct matching texts and their indices. 
    """
    output_lst = []
    match_lst = []
    source_doc = Text(source_doc)

    for input_sent_dict in input_text_lst:
        input_sent = input_sent_dict['sentence']
        if len(input_sent.split()) <= 3:
            continue
        try:
            input_sent = Text(input_sent)
            match = Matcher(input_sent, source_doc).match()
            if len(match) != 0:
                match_lst.append(input_sent_dict)
                output_dict = input_sent_dict.copy()
                output_dict['source_sentence'] = match['source_sentence']
                output_dict['score'] = match[0]['score']
                output_lst.append(output_dict)
                    
        except:
            pass

    return output_lst, match_lst
        

def get_non_direct_texts(input_text_lst, match_lst):
    """
    Returns a list of dictionaries of input document's non-direct matching indices & texts.

    Args:
        input_text_lst (list): Input document, split by sentences.
        match_ind_lst (list): Sentence indices of input documents that are flagged as direct matches.

    Returns:
        nonmatch_lst (list[dict]): Input document that is not detected as direct matches, split by sentences. Contains tuple (start_ind, end_ind, sentence).
    """
    
    nonmatch_lst = []

    for i in input_text_lst:
        if i not in match_lst:
            nonmatch_lst.append(i)

    return nonmatch_lst

def modified_output_lists(direct_output_lst, para_output_lst, threshold=0.95):
    """
    Returns the modified output lists for both direct matching and paraphrasing lists, given threshold.
    Output with similarity scores > threshold in the paraphrased output list will be moved to the direct output list.

    Args:
        direct_output_lst (list): List of dictionary of direct matching texts and their details.
        para_output_lst (list): List of dictionary of paraphrased texts and their details.
        threshold (float): Threshold of similarity score to flag paraphrased texts as direct matches.

    Returns:
        new_direct_lst (list): Modified list of dictionary of direct matching texts and their details.
        new_para_lst (list): Modified list of dictionary of paraphrased texts and their details.
    """
    filter_para_lst = []
    new_para_lst = []

    for dict in para_output_lst:
        if dict['score'] >= threshold:
            filter_para_lst.append(dict)
        else:
            new_para_lst.append(dict)

    new_direct_lst = direct_output_lst + filter_para_lst 

    return new_direct_lst, new_para_lst

def get_avg_score(input_text_lst, output_lst):
    """
    Returns the average of cosine similarity scores over number of sentences in input document.

    Args:
        input_text_lst (list): Input document, split by sentences.
        output_lst (list):  List of dictionary of matching texts and their details.

    Returns:
        avg_score (float): Average cosine similarity scores over number of sentences in input document.
    """
    input_doc_len = len(input_text_lst)
    total_score = 0
    for score in output_lst:
        total_score += score['score']
    avg_score = total_score / input_doc_len

    return avg_score

######## PARAPHRASED MATCHING FUNCTIONS ########

def load_sentbert_model(bucket, filepath, directory, sentbert_model_name):
    """
    Gets trained Sentence Transformer model based on target s3 bucket and file path.
    
    Args: 
        bucket (str): Mame of S3 bucket.
        filepath (str): Filepath of trained model in S3.
        sentbert_model_name (str): Filepath of the Sentence Transformer model.
            
    Returns:
        model (SentenceTransformer): Trained Sentence Transformer model.
    """
    s3client = boto3.client('s3')
    s3client.download_file(
        Bucket = bucket,
        Key = filepath,
        Filename = sentbert_model_name
    )
    
    tar_file = tarfile.open(sentbert_model_name)
    tar_file.extractall('./model_folder')
    tar_file.close()
    return SentenceTransformer(directory+'model_folder/trained_bert_model.h5')

def get_paraphrase_predictions(model, nonmatch_lst, source_doc, threshold):
    """
    Returns a list of json containing paraphrased sentences' details, predicted from trained Sentence Transformer model.
    
    Args: 
        model (SentenceTransformer): Sentence Transformer model.
        nonmatch_lst (list[str]): List of sentences not detected as direct matches. 
        source_doc (str): Source document.
        threshold (float): Threshold of similarity score to flag sentence as paraphrased.
            
    Returns:
        res_list (list[dict]): List of json containing paraphrased sentence details.
        example:
        [
            {
              "sentence": "\"enraged at their disappointment, Irish soldiers carved Scandinavian Stern in pieces, and is still coveted secret unrevealed.",
              "start_char_index": 2179,
              "end_char_index": 2303,
              "source_sentence": "Enraged\nat their disappointment, the Irish soldiers hewed the stern northman in pieces, and the coveted\nsecret is still unrevealed.",
              "score": 0.8140799403190613
            }
        ]
    """
    res_list = []
    try:
        source_sent = tokenize.sent_tokenize(source_doc)
        for input_sent_dict in nonmatch_lst:
            input_sent = input_sent_dict['sentence']
            if len(input_sent.split()) <= 3:
                continue

            source_embeddings = model.encode(source_sent)
            input_embeddings = model.encode(input_sent)

            res = cosine_similarity(
                    [input_embeddings],
                    source_embeddings
                )

            score = float(max(res[0]))

            if score > threshold:
                temp = input_sent_dict
                temp['source_sentence'] = source_sent[res[0].argmax()]
                temp['score'] = score
                res_list.append(temp)
    except:
        pass
                
    return res_list

def get_default_device():
    """
    Picking GPU if available or else CPU.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """
    Move tensor(s) to chosen device.
    """
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    
    return data.to(device, non_blocking=True)

######## FEATURE GENERATION FUNCTIONS ########

def get_vocab_counts(input_doc, source_doc, n):
    '''
    Using CountVectorizer, create vocab based on both texts.
    Count number of occurence of each ngram
    '''
    counts_ngram = CountVectorizer(analyzer='word', ngram_range=(n, n))
    vocab = counts_ngram.fit([input_doc, source_doc]).vocabulary_
    counts = counts_ngram.fit_transform([input_doc, source_doc])
    
    return vocab, counts.toarray()

# calculate ngram containment for each text/original file
def calc_containment(input_doc, source_doc, n):
    '''
    calculates the containment between a given text and its original text
    This creates a count of ngrams (of size n) then calculates the containment by finding the ngram count for a text file
    and its associated original tezt -> then calculates the normalised intersection
    '''
    # create vocab and count occurence of each ngram
    vocab, ngram_counts = get_vocab_counts(input_doc, source_doc, n)
    # calc containment
    intersection_list = np.amin(ngram_counts, axis = 0)
    intersection = np.sum(intersection_list)
    count_ngram = np.sum(ngram_counts[0])
    
    return intersection / count_ngram

def get_containment_scores(input_doc, source_doc, ngrams_lst):
    containment_scores = {}
    
    for ngram in ngrams_lst:
        containment = calc_containment(input_doc, source_doc, ngram)
        key_name = f"c_{ngram}"
        containment_scores[key_name] = containment
    
    return containment_scores

def get_lcm_score(input_doc, source_doc):
    max_len = max(len(input_doc), len(source_doc))
    matcher = difflib.SequenceMatcher(None, input_doc, source_doc)
    
    # calculate the ratio of the longest common subsequence and the length of the longer text
    lcs_ratio = matcher.find_longest_match(0, len(input_doc), 0, len(source_doc)).size / max_len
    
    return lcs_ratio

######## final MODEL LOADING & PREDICTION FUNCTIONS ########

def load_final_model(bucket, filepath, final_model_name):
    """
    Gets trained and finetuned final model based on target S3 bucket and file path
    
    Args: 
        bucket (str): Name of S3 bucket.
        filepath (str): Filepath of trained final model.
        final_model_name (str): File name of trained final model.
            
    Returns:
        model: Trained final model.
        
    """
    s3client = boto3.client('s3')
    s3client.download_file(
        Bucket = bucket,
        Key = filepath,
        Filename = final_model_name
    )
    
    tar_file = tarfile.open(final_model_name)
    tar_file.extractall('./model_folder')
    tar_file.close()
    final_model = joblib.load('model_folder/final_model.joblib')
    
    return final_model     

def get_feature_dict(containment_scores, lcm_score, direct_avg_score, paraphrase_avg_score):
    """
    Get a DataFrame of all features to be parsed to the trained model.

    Args:
        containment_scores (dict): A dictionary containing N n-grams containment score.
        lcm_score (float): Ratio of the longest common subsequence and the length of the longer text.
        direct_avg_score (float): Average similarity score of detected direct matching texts across the input document.
        paraphrase_avg_score (float): Average similarity score of detected paraphrased texts across the input document.

    Returns:
        feature_df (pd.DataFrame): Dataframe of all features to be parsed to the trained model (containment scores, LCM score, average cosine similarity scores from direct matching & paraphrased texts).
    """
    containment_scores['lcs_word'] = lcm_score
    containment_scores['para_detect_score'] = direct_avg_score
    containment_scores['direct_detect_score'] = paraphrase_avg_score
    feature_df = pd.DataFrame(containment_scores, index=[0])
    
    return feature_df

def get_flag_score_prediction(final_model_name, feature_df):
    """
    Returns the flag and probability predictions from the trained final model. 

    Args:
        final_model_name (final): Trained final model.
        feature_df (pd.DataFrame): Dataframe of all features to be parsed to the trained final model.

    Returns:
        plagiarism_flag (boolean): 1 means the document is plagiarised, vice-versa.
        plagiarism_scoreability (float): Probability of the document being flagged as plagiarised.
    """
    final_model = load_final_model(s3_bucket, s3_folderpath+final_model_name, final_model_name)

    plagiarism_flag = final_model.predict(feature_df)[0]
    plagiarism_score = final_model.predict_proba(feature_df)[:,1][0]
    
    return plagiarism_flag, plagiarism_score

######## FINAL OUTPUT GENERATION FUNCTIONS ########

def one_one_matching(directory, s3_bucket, s3_folderpath, sentbert_model_name, final_model_name, ngrams_lst, source_doc, source_name, input_doc, input_name):
    """
    One-to-one matching function - given 2 documents, compare and return the plagiarised flag, score and plagiarised texts.

    Args:
        directory (str): File directory.
        s3_bucket (str): Name of S3 bucket.
        s3_folderpath (str): Filepath of trained models.
        sentbert_model_name (str): Filepath of trained Sentence Transformer model.
        final_model_name (str): Filepath of trained final model.
        ngrams_lst (lst): List of selected n_grams used to generate containment scores.
        source_doc (str): Source document.
        input_doc (str): Input document.
        input_name (str): Name of input document.

    Returns:
        output_dict (dict): Dictionary containing comparison results (name of input document, plagiarised flag, score and texts).
    """
    input_text_lst = get_preprocessed_sent(input_doc)
    direct_output, match_lst = get_matching_texts(input_text_lst, source_doc)
    
    nonmatch_lst = get_non_direct_texts(input_text_lst, match_lst)
    sentence_trans_model = load_sentbert_model(s3_bucket, s3_folderpath+sentbert_model_name, directory, sentbert_model_name)
    paraphrase_output = get_paraphrase_predictions(sentence_trans_model, nonmatch_lst, source_doc, 0.7)
    
    plagiarised_text = direct_output + paraphrase_output
    plagiarised_text = sorted(plagiarised_text, key=lambda d: d['start_char_index'])

    new_direct_output, new_paraphrase_output = modified_output_lists(direct_output, paraphrase_output, 0.85)
    
    direct_avg_score = get_avg_score(input_text_lst, new_direct_output)
    paraphrase_avg_score = get_avg_score(input_text_lst, new_paraphrase_output)
    
    containment_scores = get_containment_scores(input_doc, source_doc, ngrams_lst)
    lcm_score = get_lcm_score(input_doc, source_doc)
    
    feature_df = get_feature_dict(containment_scores, lcm_score, direct_avg_score, paraphrase_avg_score)
    
    plagiarism_flag, plagiarism_score = get_flag_score_prediction(final_model_name, feature_df)
    
    output_dict = {'input_doc_name': input_name,
                   'source_doc_name': source_name,
                   'plagiarism_flag': plagiarism_flag, 
                   'plagiarism_score': plagiarism_score,
                   'plagiarised_text': plagiarised_text}
    
    return output_dict

def get_matching_output(directory, s3_bucket, s3_folderpath, sentbert_model_name, final_model_name, ngrams_lst, source_docs, input_doc, input_name):
    """
    One-to-many strings matching function - given 1 document and a list of documents, compare and return the plagiarised flag, score and plagiarised texts.

    Args:
        directory (str): File directory.
        s3_bucket (str): Name of S3 bucket.
        s3_folderpath (str): Filepath of trained models.
        sentbert_model_name (str): Filepath of trained Sentence Transformer model.
        final_model_name (str): Filepath of trained final model.
        ngrams_lst (lst): List of selected n_grams used to generate containment scores.
        source_docs (list[dict]): List of dictionaries containing source documents & source document name. E.g. [{'source_doc_name': test1, 'source_doc': teststr}]
        input_doc (str): Input document.
        input_name (str): Name of input document.

    Returns:
        output_lst (list[dict]): List of dictionaries containing comparison results (name of input document, plagiarised flag, score and texts).
    """
    output_lst = []
    for i in source_docs:
        temp_dict = one_one_matching(directory, s3_bucket, s3_folderpath, sentbert_model_name, final_model_name, ngrams_lst, i['source_doc'], i['source_doc_name'], input_doc, input_name)
        output_lst.append(temp_dict)
    
    return output_lst