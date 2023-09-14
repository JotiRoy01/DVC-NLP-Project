import logging
from tqdm import tqdm
import random
import xml.etree.ElementTree as ET #read the tags
import re


def process_posts(fd_in, fd_out_train,fd_out_test,target_tag,split) :

    line_num = 1
    for line in tqdm(fd_in) :
        try :
            fd_out = fd_out_train if random.random() > split else fd_out_test 
            attr = ET.fromstring(line).attrib #getting attributes or getting the tags
            post_id = attr.get('Id',"")
            label = 1 if target_tag in attr.get("Tags","") else 0
            title = re.sub(r"\s+"," ", attr.get("Title","")).strip() #"\s+"-> extra space
            Body = re.sub(r"\s+"," ", attr.get("Body","")).strip()
            text = f"{title} {Body}"
            
            fd_out.write(f"{post_id}\t{label}\t{text}\n")
            line_num +=1
        except Exception as e :
            msg = f"Skipping the broken line {line_num} : {e}\n"
            logging.exception(msg=msg)
