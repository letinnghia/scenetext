from utils.process import export_dataframe, export_csv, export_txt, get_file_paths, visualize_yolov5_detection
from utils.load_model import load_yolov5, load_vietocr

PATH_TO_IMAGES = r'data\input\path_to_input_folder'
PATH_TO_CSV_OUTPUT = r'data\output\csv'
PATH_TO_TXT_OUTPUT = r'data\output\txt'


if __name__ == '__main__':
    detector = load_yolov5()
    recognizer = load_vietocr()
    image_paths = get_file_paths(PATH_TO_IMAGES, '*.jpg')
    df = export_dataframe(detector, recognizer, image_paths)
    export_csv(df, PATH_TO_CSV_OUTPUT, 'final.csv')
    export_txt(r'data\output\csv\final.csv', r'data\output\txt\predicted', 0.44)