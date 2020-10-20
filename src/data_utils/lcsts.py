"""
LCSTS data preprocessing.
Author: Sarah Xu
Date: October, 2020
"""
import csv
import re
import os
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import utils

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
        self._train_txt_file_path = train_txt_file_path
        self._val_txt_file_path = val_txt_file_path
        self._test_txt_file_path = test_txt_file_path
        self._output_path = output_path
        self._training_text_csv = None
        self._training_summmary_csv = None
        self._val_text_csv = None
        self._val_summmary_csv = None
        self._test_text_csv = None
        self._test_summmary_csv = None

    @property
    def training_csv(self):
        """Get the training csv file path.

        Returns:
            str: the path for the training csv file
        """
        if self._training_text_csv is not None and self._training_summmary_csv is not None:
            return self._training_text_csv, self._training_summmary_csv
        xml_file_path = self._format_as_xml(self._train_txt_file_path, self._output_path)
        self._training_text_csv, self._training_summmary_csv, self._training_merged_csv= self._parse_xml_to_csv(xml_file_path, self._output_path, usage="train")
        return self._training_text_csv, self._training_summmary_csv

    @property
    def validation_csv(self):
        """Get the validation csv files' path.

        Returns:
            str: the path for the validation csv file
        """
        if self._val_text_csv is not None and self._val_summmary_csv is not None:
            return self._val_text_csv, self._val_summmary_csv
        xml_file_path = self._format_as_xml(self._val_txt_file_path, self._output_path)
        self._val_text_csv, self._val_summmary_csv, self._val_merged_csv= self._parse_xml_to_csv(xml_file_path, self._output_path, usage="val")
        return self._val_text_csv, self._val_summmary_csv

    @property
    def test_csv(self):
        """Get the validation csv files' path.

        Returns:
            str: the path for the validation csv file
        """
        if self._test_text_csv is not None and self._test_summmary_csv is not None:
            return self._test_text_csv, self._test_summmary_csv
        xml_file_path = self._format_as_xml(self._test_txt_file_path, self._output_path)
        self._test_text_csv, self._test_summmary_csv, self._test_merged_csv = self._parse_xml_to_csv(xml_file_path, self._output_path, usage="test")
        return self._test_text_csv, self._test_summmary_csv

    def get_random_permutation(self):
        return self._generate_rand_permutation(self.training_csv, self._output_path)

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
        # Read in the file
        with open(txt_file_path, 'r') as f:
            filedata = f.read()
        base_name = os.path.splitext(os.path.basename(txt_file_path))[0]

        # Replace <doc id=1> to <doc id="1"> to complie with xml format
        fixed = re.sub(r'<doc id=([0-9]+)>', r'<doc id="\1">', filedata)

        xml_file_path = os.path.join(output_path, base_name + '.xml')
        # Write the file out again
        with open(xml_file_path, 'w') as f:
            f.write("<?xml version=\"1.0\"?>\n<data>\n")
            f.write(fixed)
            f.write("\n</data>")
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
            str, str, str:  the short text file path
                            the summary file path
                            the merged (text + summary) file path
        """
        parser = ET.XMLParser(encoding='utf-8')
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
            print(text_dict['id'])
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
            print(merged)
        # save to csv for later processing
        # save the short text files
        text_keys = text[0].keys()
        text_file_path = os.path.join(output_path, usage + '_text.csv')
        with open(text_file_path, 'w', newline='')  as output_file:
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
        with open(merged_file_path, 'w', newline='')  as output_file:
            dict_writer = csv.DictWriter(output_file, merged_keys)
            dict_writer.writeheader()
            dict_writer.writerows(merged)
        return text_file_path, summmary_file_path, merged_file_path

    def _generate_rand_permutation(self, text_csv, output_path,
                                   output_file_name="rand_permutation.csv", num_perm=10):
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
            split_chars = utils.split_unicode_chrs(row['data'])
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_path',
                        help='Where is the training data (PART_I.txt) located?',
                        type=str,
                        default="../../data/LCSTS2.0/DATA/PART_I.txt")
    parser.add_argument('--val_path',
                        help='Where is the validation data (PART_II.txt) located?',
                        type=str,
                        default="../../data/LCSTS2.0/DATA/PART_II.txt")
    parser.add_argument('--test_path',
                        help='Where is the validation data (PART_III.txt) located?',
                        type=str,
                        default="../../data/LCSTS2.0/DATA/PART_III.txt")
    args = parser.parse_args()
    lcsts = LCSTS(args.training_path, args.val_path, args.test_path, output_path="./")
    # parse the test data and store to csv
    print("Test files saved to path {}".format(lcsts.test_csv))
