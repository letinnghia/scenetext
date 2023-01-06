import glob, os, cv2, math
from PIL import Image
import pandas as pd
import numpy as np



# get all paths of a files of the same type in a folder
def get_file_paths(path_to_folder, extension):
    file_paths = []
    for root, dirs, files in os.walk(path_to_folder):
        files = glob.glob(os.path.join(root, extension))
        for f in files:
            file_paths.append(os.path.abspath(f))

        return sorted(file_paths)



# see the cropped images from results of yolov5
def visualize_yolov5_detection(image_file_path, yolov5_model, vietocr_model):
    image = cv2.imread(image_file_path)
    df = assist_detect(yolov5_model, image, expand_up=0, expand=0)
    for i in range(df.shape[0]):
        expand_x_min = df.iloc[i]['expand_xmin']
        expand_y_min = df.iloc[i]['expand_ymin']
        expand_x_max = df.iloc[i]['expand_xmax']
        expand_y_max = df.iloc[i]['expand_ymax']

        cropped_image = image[expand_y_min : expand_y_max, expand_x_min : expand_x_max]
        cropped_image = process_image(cropped_image)
        cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        # cv2.imshow('{}'.format(i), cropped_image)
        cropped_image.show()
        print(vietocr_model.predict(cropped_image))
        cv2.waitKey(0)
        


# counting overlap area of two bounding box and eliminate one of theme if necessary (because my yolov5 model doesn't work well)
def overlap_area(coor1, coor2):
    x_min1, y_min1, x_max1, y_max1 = coor1
    x_min2, y_min2, x_max2, y_max2 = coor2
    x_center1 = (x_max1 + x_min1) / 2
    y_center1 = (y_max1 + y_min1) / 2
    x_center2 = (x_max2 + x_min2) / 2
    y_center2 = (y_max2 + y_min2) / 2
    
    overlap_area = 0
    if x_center1 >= x_center2 and y_center1 >= y_center2 and x_min1 < x_max2 and y_min1 < y_max2:
        overlap_area = (x_max2 - x_min1) * (y_max2 - y_min1)
    elif x_center1 <= x_center2 and y_center1 >= y_center2 and x_max1 > x_min2 and y_min1 < y_max2:
        overlap_area = (x_max1 - x_min2) * (y_max2 - y_min1)
    elif x_center1 <= x_center2 and y_center1 <= y_center2 and x_max1 > x_min2 and y_max1 > y_min2:
        overlap_area = (x_max1 - x_min2) * (y_max1 - y_min2)
    elif x_center1 >= x_center2 and y_center1 <= y_center2 and x_min1 < x_max2 and y_max1 > y_min2:
        overlap_area = (x_max2 - x_min1) * (y_max1 - y_min2)

    return overlap_area



# filter, expand the bounding box from yolov5's result for better vietnamese words recognition of vietocr 
def assist_detect(model, image, expand=0.1, expand_up = 0.2):
    height, width, channel = image.shape
    result = model(image)
    df = result.pandas().xyxy[0]

    if df.shape[0] == 0:
        return df

    df['xmin'] = df['xmin'].apply(lambda x: math.floor(x))
    df['ymin'] = df['ymin'].apply(lambda x: math.floor(x))
    df['xmax'] = df['xmax'].apply(lambda x: math.floor(x))
    df['ymax'] = df['ymax'].apply(lambda x: math.floor(x))
    df['expand_xmin'] = df['xmin'].copy()
    df['expand_ymin'] = df['ymin'].copy()
    df['expand_xmax'] = df['xmax'].copy()
    df['expand_ymax'] = df['ymax'].copy()
    df['dx'] = df['xmax'] - df['xmin']
    df['dy'] = df['ymax'] - df['ymin']
    df['area'] = df['dx'] * df['dy']

    drop_filter = [True for i in range(df.shape[0])]

    for i in range(df.shape[0]):
        if i == df.shape[0] - 1:
            break

        for j in range(i + 1, df.shape[0]):
            coor1 = [df.iloc[i]['xmin'], df.iloc[i]['ymin'], df.iloc[i]['xmax'], df.iloc[i]['ymax']]
            coor2 = [df.iloc[j]['xmin'], df.iloc[j]['ymin'], df.iloc[j]['xmax'], df.iloc[j]['ymax']]
            overlap = overlap_area(coor1, coor2)

            if overlap/df.iloc[i]['area'] > 0.5 or overlap/df.iloc[j]['area'] > 0.5:
                drop_filter[j] = False
            
    df = df[drop_filter]
    df = df.reset_index(drop=True)

    expand_x = np.array(df['dx']) * expand
    expand_y = np.array(df['dy']) * expand
    expand_y_top = np.array(df['dy']) * expand_up

    for i in range(df.shape[0]):
        for k in range(round(expand_x[i])):
            if df.iloc[i]['expand_xmin'] > 0:
                df.at[i, 'expand_xmin'] -= 1
            if df.iloc[i]['expand_xmax'] < width:
                df.at[i, 'expand_xmax'] += 1

        for k in range(round(expand_y[i])):
            if df.iloc[i]['expand_ymax'] < height:
                df.at[i, 'expand_ymax'] += 1

        for k in range(round(expand_y_top[i])):
            if df.iloc[i]['expand_ymin'] > 0:
                df.at[i, 'expand_ymin'] -= 1

    return df



# filter, rotate, turn cropped images to gray for better recognition of vietocr
def process_image(image):
    height, width, channels = image.shape

    if height > 2*width:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    
    cv2.threshold(cv2.GaussianBlur(image, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.threshold(cv2.bilateralFilter(image, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.threshold(cv2.medianBlur(image, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.adaptiveThreshold(cv2.GaussianBlur(image, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    cv2.adaptiveThreshold(cv2.bilateralFilter(image, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    cv2.adaptiveThreshold(cv2.medianBlur(image, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    return image



# strip, clean the output prediction from vietocr
def process_text(text):
    text = text.strip()
    text_split = text.split(' ')
    ans = ''
    for word in text_split:
        if len(word) > len(ans):
            ans = word

    return ans



# return dataframe of the final result
def export_dataframe(yolo_model, ocr_model, data_paths):
    dic = { 'file_name': [],
            'x_min': [], 'y_min': [],
            'x_max': [], 'y_max': [],
            'conf': [], 'text': [] }

    for path in data_paths:
        file_name = os.path.basename(path) 
        image = cv2.imread(path) 

        print(file_name)

        df = assist_detect(yolo_model, image, expand=0.03, expand_up=0.2) 

        for i in range(df.shape[0]):
            x_min, y_min = df.iloc[i]['xmin'], df.iloc[i]['ymin']
            x_max, y_max = df.iloc[i]['xmax'], df.iloc[i]['ymax']
            expand_x_min, expand_y_min = df.iloc[i]['expand_xmin'], df.iloc[i]['expand_ymin']
            expand_x_max, expand_y_max = df.iloc[i]['expand_xmax'], df.iloc[i]['expand_ymax']
            conf = df.iloc[i]['confidence']

            cropped_image = image[expand_y_min : expand_y_max, expand_x_min : expand_x_max]
            cropped_image = process_image(cropped_image)
            PIL_cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            text = ocr_model.predict(PIL_cropped_image)
            text = process_text(text)

            dic['file_name'].append(file_name) 
            dic['x_min'].append(x_min) 
            dic['y_min'].append(y_min) 
            dic['x_max'].append(x_max) 
            dic['y_max'].append(y_max) 
            dic['conf'].append(conf)
            dic['text'].append(text)

    return pd.DataFrame(dic)



# export csv file from dataframe
def export_csv(data_frame, path, name):
    data_frame.to_csv(path + '\\' + name, encoding='utf-8')



# create .txt files for submit
def export_txt(path_to_csv, path_to_folder_txt, conf):
    df = pd.read_csv(path_to_csv)
    for i in range(df.shape[0]):
        if df.iloc[i]['conf'] > conf:
            file_name = df.iloc[i]['file_name']
            print(file_name)
            x_min = str(df.iloc[i]['x_min'])
            y_min = str(df.iloc[i]['y_min'])
            x_max = str(df.iloc[i]['x_max'])
            y_max = str(df.iloc[i]['y_max'])
            text = df.iloc[i]['text']
            box = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max, text]
            output = ','.join([str(i) for i in box])
            with open(path_to_folder_txt + '\\' + file_name[:len(file_name) - 4] + ".txt", 'a', encoding='utf8') as f:
                f.writelines(output + "\n")
        

                
            








    










