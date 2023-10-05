import boto3

# Settings
sourceLanguage = 'en'
targetLanguage = 'it'
textString = 'This is the sixth and final post in a series on the TDWG Standards Documentation Specification (SDS).  The five earlier posts explain the history and model of the SDS, and how to retrieve the machine-readable metadata about TDWG standards.'

translate = boto3.client(
    service_name='translate', 
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='us-east-1'
)





def tranlate(sourceLanguage='en', targetLanguage='it', textString="This is Tomas Mali!"):
    result = translate.translate_text(Text=textString, SourceLanguageCode=sourceLanguage, TargetLanguageCode=targetLanguage)
    return result.get('TranslatedText')



