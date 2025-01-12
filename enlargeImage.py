import json
import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def make_square_bbox(bbox, img_width, img_height, scale=1.5):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    center_x = x1 + w // 2
    center_y = y1 + h // 2
    side_length = max(w, h) * scale

    new_x1 = center_x - side_length // 2
    new_y1 = center_y - side_length // 2
    new_x2 = center_x + side_length // 2
    new_y2 = center_y + side_length // 2

    # Ensure boundary constraints
    new_x1 = max(0, int(new_x1))
    new_y1 = max(0, int(new_y1))
    new_x2 = min(int(new_x2), img_width)
    new_y2 = min(int(new_y2), img_height)
    
    return new_x1, new_y1, new_x2, new_y2


def process_image(data):
    img_root = 'your_image_root'
    output_root = 'your_ouput_root'
    img_path = os.path.join(img_root, data['image'])
    img = cv2.imread(img_path)

    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return

    face_box = json.loads(data['box'][0])
    square_bbox = make_square_bbox(face_box, img.shape[1], img.shape[0], scale=1.5)
    sq_x1, sq_y1, sq_x2, sq_y2 = square_bbox
    face_square = img[sq_y1:sq_y2, sq_x1:sq_x2]

    output_path = os.path.join(output_root, data['image'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, face_square)


if __name__ == '__main__':
    # Input JSON file
    json_file = 'your_json_file'

    # Step 1: Load data from JSON
    try:
        with open(json_file, 'r') as f:
            faces = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {json_file}, Error: {str(e)}")
        faces = []

    # Step 2: Take the first 5000 faces
    if len(faces) > 5000:
        faces = faces[:5000]

    # Step 3: Process faces using multiprocessing
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(process_image, faces), total=len(faces), desc="Processing images"))
