from enum import Enum


class FilePathTypes(Enum):
    """ file path types for the processed csv files.
    """
    TEXT_FILE_PATH = "text_file_path"
    SUMMARY_FILE_PATH = "summmary_file_path"
    MERGED_FILE_PATH = "merged_file_path"


class UsageTypes(Enum):
    """ Usage types for lcsts data.
    """
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
