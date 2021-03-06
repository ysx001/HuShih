"""
LCSTS data preprocessing.
Author: Sarah Xu
Date: October, 2020
"""
import csv
import re
import os
import argparse
import logging
from collections import defaultdict
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import numpy as np
from data_utils.utils import split_unicode_chrs
from data_utils.constants import FilePathTypes, UsageTypes

# logging settings
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# TODO: define an abstract class for datasets
class LCSTS(object):
    """
    Read in the LCSTS2.0 dataset and perform operations on it.
    The dataset can be accessed from http://icrc.hitsz.edu.cn/Article/show/139.html.
    The data is split into three parts:
    PART_I.txt - used as training data
    PART_II.txt - used as validation data
    PART_III.txt - used as test data
    """
    def __init__(self, train_txt_file_path, val_txt_file_path,
                 test_txt_file_path, output_path="./"):
        self._input_file_paths = dict()
        self._input_file_paths[UsageTypes.TRAIN.value] = train_txt_file_path
        self._input_file_paths[UsageTypes.VALIDATION.value] = val_txt_file_path
        self._input_file_paths[UsageTypes.TEST.value] = test_txt_file_path
        self._output_file_paths = dict()
        self._output_file_paths[UsageTypes.TRAIN.value] = None
        self._output_file_paths[UsageTypes.VALIDATION.value] = None
        self._output_file_paths[UsageTypes.TEST.value] = None
        self._output_path = output_path

    @property
    def train_merged_csv(self):
        if self._output_file_paths[UsageTypes.TRAIN.value] is None:
            self.process_csv(usage=UsageTypes.TRAIN.value)
        return self._output_file_paths[UsageTypes.TRAIN.value][FilePathTypes.MERGED_FILE_PATH.value]

    @property
    def val_merged_csv(self):
        if self._output_file_paths[UsageTypes.VALIDATION.value] is None:
            self.process_csv(usage=UsageTypes.VALIDATION.value)
        return self._output_file_paths[UsageTypes.VALIDATION.value][FilePathTypes.MERGED_FILE_PATH.value]

    @property
    def test_merged_csv(self):
        if self._output_file_paths[UsageTypes.TEST.value] is None:
            self.process_csv(usage=UsageTypes.TEST.value)
        return self._output_file_paths[UsageTypes.TEST.value][FilePathTypes.MERGED_FILE_PATH.value]

    def process_csv(self, usage):
        xml_file_path = self._format_as_xml(self._input_file_paths[usage],
                                            self._output_path)
        output_dict = self._parse_xml_to_csv(xml_file_path,
                                             self._output_path,
                                             usage=usage)
        self._output_file_paths[usage] = output_dict

    def get_random_permutation(self):
        return self._generate_rand_permutation(self.training_csv,
                                               self._output_path)

    def _format_as_xml(self, txt_file_path, output_path):
        """The input file that LCSTS provides has a format that looks like xml
        but not exactly is xml.
        To use the python xml library, we need to clean it up as xml format.

        Args:
            txt_file_path (str): The location of the raw txt files
            output_path (str): The directory to save the cleaned up xml files

        Returns:
            str: The location of the output xml files.
        """
        base_name = os.path.splitext(os.path.basename(txt_file_path))[0]
        xml_file_path = os.path.join(output_path, base_name + '.xml')
        if os.path.exists(xml_file_path):
            LOG.info("File already exist %s", xml_file_path)
            return xml_file_path
        # Read in the file
        with open(txt_file_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        # process the file to fix illegal xml
        tags = ['<summary>', '<short_text>']
        filedata = []
        for i in range(len(lines)):
            if i != 0 and any(tag in lines[i-1] for tag in tags):
                # fix escape characters in content
                filedata.append(escape(lines[i]))
            elif 'doc' in lines[i]:
                # Replace <doc id=1> to <doc id="1"> to complie with xml format
                fixed = re.sub(r'<doc id=([0-9]+)>', r'<doc id="\1">', lines[i])
                filedata.append(fixed)
            else:
                filedata.append(lines[i])
        # Write the file out again
        with open(xml_file_path, 'w') as f:
            f.write("<?xml version=\"1.0\"?>\n<data>\n")
            f.writelines(filedata)
            f.write("\n</data>")
        LOG.info("Wrote sanitized xml file to %s", xml_file_path)
        return xml_file_path

    def _parse_xml_to_csv(self, xml_file_path, output_path, usage="train"):
        """
            The xml comes in with 2 formats.
            For PART_I (Training), the data is
                <doc id=x>
                    <summary>
                        ...
                    </summary>
                    <short_text>
                        ...
                    </short_text>
                </doc>
            For PART_II and PART_III (Val and Test), the data is
                <doc id=0>
                    <human_label>5</human_label>
                    <summary>
                        ...
                    </summary>
                    <short_text>
                        ...
                    </short_text>
                </doc>
        Args:
            xml_file_path (str): The input of the xml file.
            output_path (str): The directory to save the output files.
            usage (str, optional): The usage of the data e.g. train, eval, test.
                                   Defaults to "training".

        Returns:
            dict:  a dict of processed file path
        """
        merged_file_path = os.path.join(output_path, usage + '_merged.csv')
        file_path_dict = defaultdict(str)
        if os.path.exists(merged_file_path):
            file_path_dict[FilePathTypes.MERGED_FILE_PATH.value] = merged_file_path
            LOG.info("File already exist %s", merged_file_path)
            return file_path_dict
        with open(xml_file_path, "r") as f:
            # Read each line in the file, readlines() returns a list of lines
            content = f.readlines()
            # Combine the lines in the list into a string
            content = "".join(content)
            soup = BeautifulSoup(content, 'lxml-xml')
        LOG.info("Parsing xml file %s", xml_file_path)
        merged = []
        docs = soup.find_all('doc')
        for doc in docs:
            merged_dict = {}
            if usage != UsageTypes.TRAIN.value:
                human_label = int(doc.human_label.get_text().strip())
                if human_label < 3:
                    continue
                merged_dict['human_label'] = human_label
            merged_dict['id'] = int(doc.get('id'))
            merged_dict['summary'] = doc.summary.get_text().strip()
            merged_dict['short_text'] = doc.short_text.get_text().strip()
            merged.append(merged_dict)
        merged_keys = merged[0].keys()
        with open(merged_file_path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, merged_keys)
            dict_writer.writeheader()
            dict_writer.writerows(merged)
        file_path_dict[FilePathTypes.MERGED_FILE_PATH.value] = merged_file_path
        LOG.info("Wrote parsed csv file to %s", file_path_dict)
        return file_path_dict

    def _parse_xml_to_csv_old(self, xml_file_path, output_path, usage="train"):
        """
            The xml comes in with 2 formats.
            For PART_I (Training), the data is
                <doc id=x>
                    <summary>
                        ...
                    </summary>
                    <short_text>
                        ...
                    </short_text>
                </doc>
            For PART_II and PART_III (Val and Test), the data is
                <doc id=0>
                    <human_label>5</human_label>
                    <summary>
                        ...
                    </summary>
                    <short_text>
                        ...
                    </short_text>
                </doc>
        Args:
            xml_file_path (str): The input of the xml file.
            output_path (str): The directory to save the output files.
            usage (str, optional): The usage of the data e.g. train, eval, test.
                                   Defaults to "training".

        Returns:
            str, str, str:  the short text file path
                            the summary file path
                            the merged (text + summary) file path
        """
        parser = ET.XMLParser(encoding='utf-8', recover=True)
        # create element tree object
        tree = ET.parse(xml_file_path, parser=parser)
        # get root element
        root = tree.getroot()
        text = []  # list of dict for all the short_text data
        summary = []  # list of dict for all the summaries
        merged = []
        for doc in root:
            text_dict = {}
            summary_dict = {}
            merged_dict = {}
            text_dict['id'] = int(doc.attrib['id'])
            summary_dict['id'] = int(doc.attrib['id'])
            merged_dict['id'] = int(doc.attrib['id'])
            for item in doc:
                if item.tag == "short_text":
                    text_dict[item.tag] = item.text.strip()
                    merged_dict[item.tag] = item.text.strip()
                elif item.tag == "summary":
                    summary_dict[item.tag] = item.text.strip()
                    merged_dict[item.tag] = item.text.strip()
                elif item.tag == "human_label":
                    summary_dict[item.tag] = int(item.text.strip())
                    text_dict[item.tag] = int(item.text.strip())
                    merged_dict[item.tag] = int(item.text.strip())
            text.append(text_dict)
            summary.append(summary_dict)
            merged.append(merged_dict)
        # save to csv for later processing
        # save the short text files
        text_keys = text[0].keys()
        text_file_path = os.path.join(output_path, usage + '_text.csv')
        with open(text_file_path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, text_keys)
            dict_writer.writeheader()
            dict_writer.writerows(text)
        # save the summary files
        summary_keys = summary[0].keys()
        summmary_file_path = os.path.join(output_path, usage + '_summary.csv')
        with open(summmary_file_path, 'w', newline='')  as output_file:
            dict_writer = csv.DictWriter(output_file, summary_keys)
            dict_writer.writeheader()
            dict_writer.writerows(summary)
        # save the merged files
        merged_keys = merged[0].keys()
        merged_file_path = os.path.join(output_path, usage + '_merged.csv')
        with open(merged_file_path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, merged_keys)
            dict_writer.writeheader()
            dict_writer.writerows(merged)
        file_path_dict = defaultdict(str)
        file_path_dict[FilePathTypes.TEXT_FILE_PATH.value] = text_file_path
        file_path_dict[FilePathTypes.SUMMARY_FILE_PATH.value] = summmary_file_path
        file_path_dict[FilePathTypes.MERGED_FILE_PATH.value] = merged_file_path
        return file_path_dict

    def _generate_rand_permutation(self, text_csv, output_path,
                                   output_file_name="rand_permutation.csv",
                                   num_perm=10):
        """Generate random permutations of data given text csv file.

        Args:
            text_csv (str): the file path for text csv
            output_path (str): The directory to save the output files.
            output_file_name (str, optional): The file name fore the output csv.
                                              Defaults to "rand_permutation.csv".
            num_perm (int, optional): The number of random permutations. Defaults to 10.

        Returns:
            str: The location of the randomly permutated sentences.
        """
        with open(text_csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            original_data = list(reader)
        new_data = []
        for row in original_data:
            # print(row['id'], row['data'])
            raw = {}
            raw['data'] = row['data']
            raw['label'] = 1
            new_data.append(raw)
            split_chars = split_unicode_chrs(row['data'])
            for _ in range(num_perm):
                rand = {}
                rand['data'] = "".join(np.random.permutation(split_chars))
                rand['label'] = 0
                new_data.append(rand)
        # write to file
        rand_perm_file_path = os.path.join(output_path, output_file_name)
        rand_perm_keys = new_data[0].keys()
        with open(rand_perm_file_path, 'w', newline='')  as output_file:
            dict_writer = csv.DictWriter(output_file, rand_perm_keys)
            dict_writer.writeheader()
            dict_writer.writerows(new_data)
        return rand_perm_file_path
