FROM public.ecr.aws/lambda/python:3.8

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY ./app ${LAMBDA_TASK_ROOT}
COPY ./app/nltk_data ./nltk_data

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "plagiarism_detector.lambda_handler" ]