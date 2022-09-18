import json
import configparser

if __name__ == '__main__':
    output_path = './data/MuPoTS/MuPoTS-3D.json'
    img_path = './data/MuPoTS/MultiPersonTestSet/MultiPersonTestSet'
    with open(output_path, 'r') as f:
        output = json.load(f)
    image = output['images']
    ann = output['annotations']

    m =0
    for n in range(len(image)):
        conf = configparser.ConfigParser()
        image_intrinsic = image[n]['intrinsic']
        conf.add_section('intrinsic')  # 添加section
        conf.set('intrinsic', 'intrinsic', str(image_intrinsic))

        image_id = image[n]['id']
        count = 1
        conf.add_section('keypoints_cam')  # 添加section
        conf.add_section('keypoints_img')  # 添加section
        for m in range(len(ann)):
            if ann[m]['image_id'] == image_id:
                keypoints_cam = ann[m]['keypoints_cam']
                conf.set('keypoints_cam', 'keypoints_cam'+str(count), str(keypoints_cam))
                keypoints_img = ann[m]['keypoints_img']
                conf.set('keypoints_img', 'keypoints_img'+str(count), str(keypoints_img))
                count += 1
            m += 1
        conf.add_section('image_id')  # 添加section
        conf.set('image_id', 'image_id', str(image_id))
        file_name = image[n]['file_name']
        save_path = img_path + '/' + file_name[:-4] + '.ini'
        with open(save_path, 'w', encoding='utf-8') as f:
            conf.write(f)
        print('have made' + str(n))
