from flask import Flask,request,jsonify
from flask_cors import CORS
import spacy
import en_core_web_sm
import pandas as pd
import re
import pickle
import random
from random import randint
from pandas import datetime
from datetime import timedelta
from spacy.matcher import PhraseMatcher
from spacy.tokenizer import Tokenizer
from PIL import Image
from datetime import datetime
import requests
from io import BytesIO
import cv2
import numpy as np
import pytesseract
import io
from wand.image import Image as wi
import urllib
from PyPDF2 import PdfFileReader, PdfFileWriter
from pdf2image import convert_from_path
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

"""request_data = {
    "files":["fgfhgh,fdgh"],
    "description":" Niraj, Vivek you both have to run daily for 1-2 hrs",
    "injection":"tetnus ",
    "medicine":"Vivek Sharma you should drink 1-2l water daily",
    "labReport":"Abhinav you have AIDS"
    }"""



app = Flask(__name__)
CORS(app)



# ** Function to extract regex patterns from given string **

def extract_regex(string, doc, og_string):
    a = []
    expression = string
    for match in re.finditer(expression, doc.text):
        start,end = match.span()
        span = doc.char_span(start,end)
        l = []
        l.append(og_string[start:end])
        l.append(start)
        l.append(end)
        a.append(l)
    return a

# ** Inputs **
#    -> Input string i.e Original String
#    -> Trained en_core_web_sm spacy model
#    -> Blank spacy model
#    -> Choices(2)
#       -> 1. To remove dates completely from the EHR
#       -> 2. To shift dates present in the EHR to have a time domain for research purposes

def deidentifier_func(input_string, nlp_trained_model, nlp_blank_model, choice):
    #doc = nlp_trained_model((open(input_string)).read())
    doc = nlp_trained_model(input_string)
    #original_string = open((input_string)).read()
    original_string = input_string



    # ** Calling extract_regex function to get list of all the matched regex pattern **
    date_list = extract_regex(r"\D([0-9]{4}|[0-9]{1,2})(\/|-)[0-9]{1,2}(\/|-)([0-9]{1,2}|[0-9]{4})\D",
                              doc,original_string)

    for i in range(len(date_list)):
        date_list[i][1] = date_list[i][1] + 1
        date_list[i][2] = date_list[i][2] - 1
        date_list[i][0] = original_string[date_list[i][1] : date_list[i][2]]

    # ** For choice 1 **
    """if(choice == 1):
        for a in date_list:
            count = 0
            for i in range(a[1], a[1] + 4):
                if(original_string[i].isnumeric()):
                    count = count + 1
            if(count == 4):
                original_string=original_string[:a[1]+4]+''*(a[2]-a[1]-4)+original_string[a[2]:]
            else:
                count = 0
                for j in range(a[2], a[2]-5, -1):
                    if(original_string[j].isnumeric()):
                        count = count + 1
                if(count == 4):
                    original_string=original_string[:a[1]]+''*(a[2]-a[1]-4)+original_string[a[2]-4:]
                elif(count == 3):
                    original_string=original_string[:a[1]]+''*(a[2]-a[1]-2)+original_string[a[2]-2:]
                else:
                    original_string=original_string[:a[1]]+''*(a[2]-a[1])+original_string[a[2]:]

    """
    # ** For Choice 2 **
    date_shift = []
    temp_1 = 0
    temp_2 = 0
    random_value = randint(0, 90)
    if(choice == 2):
        for temp in range(len(date_list)):
            temp_list = []
            text = date_list[temp][0]
            start = date_list[temp][1] + temp_2
            end = date_list[temp][2] + temp_2
            # Converting dates to pandas datetime so as to use timedelta function
            pandas_date = pd.to_datetime(text,
                                         infer_datetime_format= True,
                                         errors= 'ignore')
            if(type(pandas_date) != str):
                pandas_date = pandas_date + timedelta(days = random_value)
                original_string = original_string[:start]+str(pandas_date)[:-9]+original_string[end:]
                temp_2 = temp_2 + (len(str(pandas_date)[:-9])-len(text))
                temp_list.append(str(pandas_date)[:-9])
                temp_list.append(start)
                temp_list.append(start + len(str(pandas_date)[:-9]))
                date_shift.append(temp_list)




    # ** Extracting all various identifiers using regex pattern **
    #dob_list = extract_regex(r"^(0[1-9]|1[012])[-/.](0[1-9]|[12][0-9]|3[01])[-/.](19|20)\\d\\d$",
    #                         doc, original_string)

    aadhar_list = extract_regex(r"(\d{4}(\s|\-)\d{4}(\s|\-)\d{4})",doc,original_string)

    ssn_list = extract_regex(r"^\d{9}$", doc, original_string)

    mail_list=extract_regex(r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*)@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])",
                              doc,original_string)

    ip_list=extract_regex(r"((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
                            doc,original_string)

    # ** Now de-identifying them **
    #for a in dob_list:
    #    original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]

    for a in aadhar_list:
        original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]

    for a in ssn_list:
        original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]

    for a in mail_list:
        original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]

    for a in ip_list:
        original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]



    # ** Now to extract urls and licence plate numbers from last updated original_string
    #    and then deidentifying them too **
    doc = nlp_trained_model(original_string)
    url_list=extract_regex(r"(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?",
                             doc,original_string)

    license_plate_list=extract_regex(r"[A-Z]{2}[ -][0-9]{1,2}(?: [A-Z])?(?: [A-Z]*)? [0-9]{4}",
                                       doc,original_string)

    for a in ip_list:
        original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]

    for a in ip_list:
        original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]



    # ** Now to extract contact details i.e phone numbers and fax numbers from last updated
    #    original_string and then deidentifying them too **
    doc = nlp_trained_model(original_string)
    #indian_ph_no = extract_regex(r"((\+*)((0[ -]+)*|(91 )*)(\d{12}+|\d{10}+))|\d{5}([- ]*)\d{6}",
    #                               doc, original_string)

    usa_ph_no = extract_regex(r"^(\([0-9]{3}\) |[0-9]{3}-)[0-9]{3}-[0-9]{4}$",
                                doc, original_string)

    phone_fax_list1=extract_regex(r"(?:(?:(?:(\+)((?:[\s.,-]*[0-9]*)*)(?:\()?\s?((?:[\s.,-]*[0-9]*)+)(?:\))?)|(?:(?:\()?(\+)\s?((?:[\s.,-]*[0-9]*)+)(?:\))?))((?:[\s.,-]*[0-9]+)+))",
                                  doc,original_string)

    phone_fax_list2=extract_regex(r"\D(\+91[\-\s]?)?[0]?(91)?[789]\d{9}\D",
                                  doc,original_string)

    for i in range(len(phone_fax_list2)):
        phone_fax_list2[i][1]=phone_fax_list2[i][1]+1
        phone_fax_list2[i][2]=phone_fax_list2[i][2]-1
        phone_fax_list2[i][0]=original_string[phone_fax_list2[i][1]:phone_fax_list2[i][2]]

    phone_fax_list=[]
    for a in phone_fax_list1:
        phone_fax_list.append(a)
    for a in phone_fax_list2:
        phone_fax_list.append(a)

    for a in phone_fax_list1:
        original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]

    for a in phone_fax_list2:
        original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]

    #for a in indian_ph_no:
    #    original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]

    for a in usa_ph_no:
        original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]



    # ** Extracting account details and other identification details and deidentifying them**
    doc = nlp_trained_model(original_string)

    pan_list = extract_regex(r"[A-Z]{5}\d{4}[A-Z]{1}",doc,original_string)

    passport_list = extract_regex(r"[A-Z]{1}\d{7}",doc,original_string)

    account_and_serial_list=extract_regex(r"\d{9,18}",doc,original_string)

    credit_card_list=extract_regex(r"\d{5}(\s|\-)\d{5}(\s|\-)\d{5}|\d{4}(\s|\-)\d{4}(\s|\-)\d{4}(\s|\-)\d{4}",
                                     doc,original_string)

    for a in pan_list:
        original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]

    for a in passport_list:
        original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]

    for a in account_and_serial_list:
        original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]

    for a in credit_card_list:
        original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]


    # ** Extracting MRN(Medical Report Number) if present and assumning it to be 7 digit**
    doc = nlp_trained_model(original_string)
    mrn_list = extract_regex(r"\d{7}", doc, original_string)

    for a in mrn_list:
        original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]


    # Now we've deidentified all the details except address



    # ** For extracting address we use a list of address_identifiers for addresses smaller
    #    than street names and match them with every element in spacy doc object.
    #    Matched object are then added to our address_list **

    address_identifier=['st','niwas','aawas','palace','road','block','gali','sector',
                        'flr','floor','path','near','oppo','bazar','house','nagar',
                        'bypass','bhawan','street','rd','sq','flat','lane','gali',
                        'circle','bldg','ave','mandal','avenue','tower','nagar','marg',
                        'chowraha','lane','heights','plaza','park','garden','gate','villa',
                        'market','apartment','chowk']

    doc = nlp_trained_model(original_string)
    address_list = []

    for i in doc:
        if(len(i)>1 and '\n' not in str(i)):
            if(str(i).lower() in address_identifier):
                address_list.append(i)

    # ** Now to remove the identified addresses after getting their position in og_string
    address_index = []
    temp_2 = 0
    length = len(original_string)
    for i in address_list:
        while(1):
            index = original_string.find(str(i), temp_2, length)
            if(index == -1):
                break
            if(index!=0 and index!=length):
                if((original_string[index-1].isalpha() or
                    original_string[index+len(str(i))].isalpha())):
                    temp_2 = index + len(str(i))
                else:
                    break
        address_index.append(index)
        temp_2 = index + len(str(i))

    temp_1 = 0
    new_address_list = []
    if(address_index != []):
        temp_1 = address_index[0]
        a = []
        for b in address_index:
            if(b-temp_1 < 20):
                a.append(b)
                temp_1 = b
            else:
                new_address_list.append(a)
                a = []
                a.append(b)
                temp_1 = b
        new_address_list.append(a)


    # ** Removing the complete word in which the addres_identifier was used **
    addr_list = []
    for a in new_address_list:
        flag = []
        j = a[0]
        while(j!=-1 and original_string[j] not in [',','\n','.',';']):
            j = j-1
        startt = j
        index_1 = startt
        count = 8
        while(count and j!=-1 and original_string[j]!= '\n'):
            if(original_string[j].isdigit()):
                startt = j
            j = j-1
            count = count - 1
        j = a[-1]
        #print(j)
        while(j!=-1 and original_string[j] not in [',','\n','.',';']):
            j = j+1
        endd = j
        index_2 = endd
        count = 7
        while(count and j!=length and original_string[j]!='\n'):
            if(original_string[j].isdigit()):
                endd = j
            j = j+1
            count = count-1

        if((original_string[index_1]!='.' or original_string[index_2]!='.') and (index_2-index_1)<50):
            if(original_string[startt] == '\n'):
                startt = startt+1
            if(original_string[endd] == '\n'):
                endd = endd-1
            flag.append(original_string[startt:endd+1])
            flag.append(startt)
            flag.append(endd)
            addr_list.append(flag)


    for a in addr_list:
        original_string = original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]
    
    # ** After deidentifying all these details we are now left with only names, dates, age
    #    which cannot be identified by regular expression **

    # To extract dates we use spacy's pre-trained en_core_web_sm model along with
    # some modifications to the default model according to our requirements

    time_identifier=['YEAR', 'YEARS', 'AGE', 'AGES', 'MONTH', 'MONTHS', 'DECADE',
                    'CENTURY','WEEK','DAILY', 'DAY', 'DAYS', 'NIGHT',
                    'NIGHTS', 'WEEKLY', 'MONTHLY', 'YEARLY']

    doc_1 = nlp_trained_model(original_string)
    new_date_list = []
    for entities in doc_1.ents:
        if(str(entities.text).count('X')<2):
            date = []
            if(entities.label_ == 'DATE' and
               (sum([True if i not in
                     original_string[entities.start_char:entities.end_char].upper()
                     else False for i in time_identifier]) == len(time_identifier)) and
               (entities.end_char-entities.start_char)>4 and
               sum(c.isdigit() for c in original_string[entities.start_char:entities.end_char])>=1 and
               sum(c.isalpha() for c in original_string[entities.start_char:entities.end_char])>=1):
                date.append(entities.text)
                date.append(entities.start_char)
                date.append(entities.end_char)
                new_date_list.append(date)


    for a in new_date_list:
        count = 0
        for i in range(a[1], a[1] + 4):
            if(original_string[i].isnumeric()):
                count = count + 1
        if(count == 4):
            original_string=original_string[:a[1]+4]+'X'*(a[2]-a[1]-4)+original_string[a[2]:]
        else:
            count = 0
            for j in range(a[2], a[2]-5, -1):
                if(original_string[j].isnumeric()):
                    count = count + 1
                if(count == 4):
                    original_string=original_string[:a[1]]+'X'*(a[2]-a[1]-4)+original_string[a[2]-4:]
                elif(count == 3):
                    original_string=original_string[:a[1]]+'X'*(a[2]-a[1]-2)+original_string[a[2]-2:]
                else:
                    original_string=original_string[:a[1]]+'X'*(a[2]-a[1])+original_string[a[2]:]

    final_date_list = []
    if(choice == 1):
        for a in new_date_list:
            final_date_list.append(a)
        for a in new_date_list:
            final_date_list.append(a)

    # final_date_list contains all the dates we extracted including regex and spacy model


    # ** Now going for age part, we use the spacy's phrasematcher
    #    which takes input as patterns we want to match and
    #    outputs the start and end index of matched pattern **

    try:
        age_list = []
        matcher = PhraseMatcher(nlp_trained_model.vocab, attr="SHAPE")
        age_identifier = ['YEAR', 'YEARS', 'Y/O', 'AGES', 'AGE', 'Y.O',
                          'Y.O.','AGED','AGE IS']
        matcher.add("age",None,nlp_blank_model("76 year old"),nlp_blank_model("aged 58"),
                    nlp_blank_model('aged 123'),nlp_blank_model("54 y/o"),
                    nlp_blank_model("age is 59"),nlp_blank_model("123 y/o"),
                    nlp_blank_model("ages 35"),nlp_blank_model("age 45"),
                    nlp_blank_model("ages 123"),nlp_blank_model("age 123"),
                    nlp_blank_model("54 years old"),nlp_blank_model("124 years old"),
                    nlp3("41 y.o."),nlp_blank_model("123 y.o."),
                    nlp_blank_model('113 year old'))

        doc = nlp_blank_model(original_string)
        for match_id, start, end in matcher(doc):
            if(sum([True if i in str(doc[start:end]).upper()
                   else False for i in age_identifier]) >= 1):
                a = []
                for i in range(start, end):
                    if(str(doc[i:i+1]).isnumeric()):
                        if(int(str(doc[i:i+1])) > 89):
                            result = st.find(str(doc[start:end]))
                            count = 0
                            for j in range(result,result.len(str(doc[start:end]))):
                                if(original_string[j:j+1].isnumeric() and count==0):
                                    sstart = j
                                if(original_string[j:j+1].isnumeric()):
                                    count = count+1
                            a.append(original_string[sstart:sstart+count])
                            a.append(sstart)
                            a.append(sstart + count)
                            age_list.append(a)
                            original_string = original_string[:sstart]+'X'*count+original_string[sstart+count:]
    except:
        None




    # ** Last step is packing all the extracted pattern in a dict
    info_dict = {}
    info_dict['date'] = final_date_list
    #info_dict['dob'] = dob_list
    info_dict['aadhar'] = aadhar_list
    info_dict['ssn'] = ssn_list
    info_dict['mail'] = mail_list
    info_dict['ip'] = ip_list
    info_dict['url'] = url_list
    info_dict['licence_plate'] = license_plate_list
    #info_dict['indian_ph_no'] = indian_ph_no
    info_dict['usa_ph_no'] = usa_ph_no
    info_dict['phone_fax'] = phone_fax_list
    info_dict['pan'] = pan_list
    info_dict['passport'] = passport_list
    info_dict['account_details'] = account_and_serial_list
    info_dict['credit_card'] = credit_card_list
    info_dict['age'] = age_list
    info_dict['address'] = addr_list
    info_dict['medical_report_no'] = mrn_list
    info_dict['date_shift'] = date_shift

    shift = random_value

    if(choice == 1):
        return(original_string, info_dict, None)
    else:
        return(original_string, info_dict, shift)


# ** Creating our EHR function that manages de-identification function by taking inputs **
#   -> String
#   -> Choices(2)
#       -> 1. To remove dates completely from the EHR
#       -> 2. To shift dates present in the EHR to have a time domain for research purposes

def EHR_data_extractor(input_string):

    # ** We've used 2 pickle files as a lookup table to reduce error **
    #    -> Containing all the medical fields
    #    -> Containing names of cities and states of India

    with open('medical_fields.pkl', 'rb') as file:
        medical_field_data = pickle.load(file)

    with open('city_state_of_india.pkl', 'rb') as file:
        city_state_list_data = pickle.load(file)


    # ** Loading spacy's pre trained model **
    nlp_trained_model = en_core_web_sm.load()


    # ** Now loading a en_core_web_sm model
    #    but it is a re-trained spacy language model on medical data **
    nlp_re_trained_model = spacy.load('trained_spacy_model_on_medical_data')
    #nlp_re_trained_model = spacy.load('last_modified_model')


    # ** Loading a blank spacy model **
    nlp_blank_model = spacy.blank('en')

    choice = 2
    processed_string = input_string
    # Calling the de-identifier function
    processed_string, dictionary, date_shift = deidentifier_func(input_string,
                                                                 nlp_trained_model,
                                                                 nlp_blank_model,
                                                                 choice)

    # Now to extract names of PERSON & ORG from the processed_string we use our
    # re-trained spacy model on medical data and along with it we de-identify them too

    nlp_tokenizer = Tokenizer(nlp_blank_model.vocab)
    doc = nlp_re_trained_model(processed_string)
    person_org_list = []

    for entities in doc.ents:
        if(str(entities.text).count('X') < 2):
            tokens = nlp_tokenizer(str(entities.text))
            if(sum([True if str(i).lower() in medical_field_data or
                    '\n' in str(i) or
                    str(i).lower() in city_state_list_data
                    else False for i in tokens]) != len(tokens)):
                pre_list_for_p_org = []
                pre_list_for_p_org.append(entities.text)
                pre_list_for_p_org.append(entities.start_char)
                pre_list_for_p_org.append(entities.end_char)
                person_org_list.append(pre_list_for_p_org)

    #dictionary = {}
    dictionary['person_&_org'] = person_org_list

    # Now de-identifying it by 'X'
    for a in person_org_list:
        processed_string= processed_string[:a[1]]+'X'*(a[2] - a[1])+processed_string[a[2]:]

    # ** Returning processed_string, dictionary and date_shift **
    return processed_string


# Functions to deidentify images
def url_to_img(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def img_preprocessing(url):
    # Getting the image from the url provided
    image = url_to_img(url)
    
    # Rescaling the image
    image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    # Converting it to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Now applying dilation and erosion to remove noise
    kernel = np.ones((1,1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    
    # Blurring to smooth out the edges
    image = cv2.GaussianBlur(image, (5,5), 0)
    
    # Applying threshold to get only b&w image (Binarization) *MUST*
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return image

def img_OCR(image):
    #preprocessing it
    image = img_preprocessing(image)
    
    return pytesseract.image_to_string(image, lang='eng')

def img_deidentifier(url):
    
    img = url_to_img(url)
    input_string = img_OCR(img)
    processed_string = EHR_data_extractor(input_string)
    
    now = datetime.now()
    deid_img_path = "static/"+"deidentified_"+now.strftime("%d-%m-%Y_%H:%M:%S:%f")+".txt"
    
    with open(deid_img_path, "a") as f:
        f.write(processed_string)
    
    return deid_img_path

# Functions to deidentify PDFs
def url_to_pdf(url):
    response = requests.get(url, stream=True)
    #print(response)
    #print(type(response))
    #print(response.content)
    now = datetime.now()
    pdf_file_path = "static/"+now.strftime("%d-%m-%Y %H:%M:%S:%f")+".pdf"
    
    with open(pdf_file_path, 'wb') as f:
        f.write(response.content)
    
    return pdf_file_path

def pdf_OCR(pdf_file_path):
    pages = convert_from_path(pdf_file_path, 500)
    
    # To store images of each page of PDF to image
    image_counter = 1
    now = datetime.now()
    for page in pages:
        filename = "/home/amit_bahir/Desktop/deidentification/reports/page_" + now.strftime("%d-%m-%Y_%H:%M:%S:%f_") + str(image_counter) + ".jpg"
        page.save(filename, 'JPEG')
        image_counter = image_counter + 1
    
    filelimit = image_counter - 1
    
    txt_file_path = "static/"+"to_be_deidentified" + now.strftime("%d-%m-%Y_%H:%M:%S:%f")+".txt"
    
    with open(txt_file_path, "a") as f:
        for i in range(1, filelimit + 1):
            filename = "/home/amit_bahir/Desktop/deidentification/reports/page_" + now.strftime("%d-%m-%Y_%H:%M:%S:%f_") + str(i) + ".jpg"
            text = str(((pytesseract.image_to_string(Image.open(filename)))))
            text = text.replace('-\n', '')
            f.write(text)
    return txt_file_path

def pdf_deidentifier(url):
    
    pdf_file_path = url_to_pdf(url)
    txt_file_path = pdf_OCR(pdf_file_path)
    
    with open(txt_file_path, "r") as f:
        input_string = f.read()
    
    processed_string = EHR_data_extractor(input_string)
    
    now = datetime.now()
    deid_pdf_path = "static/"+"deidentified_"+now.strftime("%d-%m-%Y_%H:%M:%S:%f")+".txt"
    
    with open(deid_pdf_path, "a") as f:
        f.write(processed_string)
        
    return deid_pdf_path



@app.route("/ml", methods=["POST"])  # Creating a decorator
def deidentification():


    #return EHR_data_extractor(str(data["diagnosis"]))
    #return data["diagnosis"]
    if request.method == 'POST':
        request_data = request.get_json()
        #return (request_data["diagnosis"])
        string = request_data['description']+".\n"+request_data['medicine']+".\n"+request_data['injection']+".\n"+request_data['labReport']+".\n"
        deidentified_string = EHR_data_extractor(string)
        #print(deidentified_string)
        deidentified_list = deidentified_string.splitlines()
        #print(deidentified_list)

        # Deidentifying files
        deidentified_files = []
        for file in request_data['files']:
            identifier = file[-3:]
            if identifier == "png":
                deid_img_path = img_deidentifier(file)
                deid_img_path = "http://192.168.43.49:5000/" + deid_img_path
                deidentified_files.append(deid_img_path)
            elif identifier == "jpg":
                deid_img_path = img_deidentifier(file)
                deid_img_path = "http://192.168.43.49:5000/" + deid_img_path
                deidentified_files.append(deid_img_path)
            elif identifier == "pdf":
                deid_pdf_path = pdf_deidentifier(file)
                deid_pdf_path = "http://192.168.43.49:5000/" + deid_pdf_path
                deidentified_files.append(deid_pdf_path)
            elif identifier == "txt":
                with open(file, "r") as f:
                    input_string = f.read()
                processed_string = EHR_data_extractor(input_string)
                now = datetime.now()
                deid_text_path = "static/"+"deidentified_"+now.strftime("%d-%m-%Y %H:%M:%S:%f")+".txt"
                with open(deid_text_path, "a") as f:
                    f.write(processed_string)
                deidentified_files.append(deid_text_path)

        # Now deidentifying fields
        deidentified_data = {
            'description' :deidentified_list[0], #EHR_data_extractor(request_data['description']),
            'medicine' : deidentified_list[1], #EHR_data_extractor(request_data['medicine']),
            'injection' : deidentified_list[2],#EHR_data_extractor(request_data['injection']),
            'labReport' : deidentified_list[3],#EHR_data_extractor(request_data['labReport']),
            'files' : deidentified_files #request_data['files']
        }
        return jsonify(deidentified_data)

    return jsonify({"message" : "Didn't perform de_identification"})

"""
@app.route("/test")
def g():
    return "this works"
"""


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port = 5000)
