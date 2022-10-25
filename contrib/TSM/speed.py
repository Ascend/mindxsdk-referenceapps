# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import subprocess
import shutil
from PIL import Image
import mindx.sdk as sdk
import numpy as np

file_path = "./model/TSM.om"
device_id = 0 
cls_text = ['abseiling', 'air drumming', 'answering questions', 'applauding', 'applying cream', 'archery', 'arm wrestling', 'arranging flowers', 'assembling computer', 'auctioning', 'baby waking up', 'baking cookies', 'balloon blowing', 'bandaging', 'barbequing', 'bartending', 'beatboxing', 'bee keeping', 'belly dancing', 'bench pressing', 'bending back', 'bending metal', 'biking through snow', 'blasting sand', 'blowing glass', 'blowing leaves', 'blowing nose', 'blowing out candles', 'bobsledding', 'bookbinding', 'bouncing on trampoline', 'bowling', 'braiding hair', 'breading or breadcrumbing', 'breakdancing', 'brush painting', 'brushing hair', 'brushing teeth', 'building cabinet', 'building shed', 'bungee jumping', 'busking', 'canoeing or kayaking', 'capoeira', 'carrying baby', 'cartwheeling', 'carving pumpkin', 'catching fish', 'catching or throwing baseball', 'catching or throwing frisbee', 'catching or throwing softball', 'celebrating', 'changing oil', 'changing wheel', 'checking tires', 'cheerleading', 'chopping wood', 'clapping', 'clay pottery making', 'clean and jerk', 'cleaning floor', 'cleaning gutters', 'cleaning pool', 'cleaning shoes', 'cleaning toilet', 'cleaning windows', 'climbing a rope', 'climbing ladder', 'climbing tree', 'contact juggling', 'cooking chicken', 'cooking egg', 'cooking on campfire', 'cooking sausages', 'counting money', 'country line dancing', 'cracking neck', 'crawling baby', 'crossing river', 'crying', 'curling hair', 'cutting nails', 'cutting pineapple', 'cutting watermelon', 'dancing ballet', 'dancing charleston', 'dancing gangnam style', 'dancing macarena', 'deadlifting', 'decorating the christmas tree', 'digging', 'dining', 'disc golfing', 'diving cliff', 'dodgeball', 'doing aerobics', 'doing laundry', 'doing nails', 'drawing', 'dribbling basketball', 'drinking', 'drinking beer', 'drinking shots', 'driving car', 'driving tractor', 'drop kicking', 'drumming fingers', 'dunking basketball', 'dying hair', 'eating burger', 'eating cake', 'eating carrots', 'eating chips', 'eating doughnuts', 'eating hotdog', 'eating ice cream', 'eating spaghetti', 'eating watermelon', 'egg hunting', 'exercising arm', 'exercising with an exercise ball', 'extinguishing fire', 'faceplanting', 'feeding birds', 'feeding fish', 'feeding goats', 'filling eyebrows', 'finger snapping', 'fixing hair', 'flipping pancake', 'flying kite', 'folding clothes', 'folding napkins', 'folding paper', 'front raises', 'frying vegetables', 'garbage collecting', 'gargling', 'getting a haircut', 'getting a tattoo', 'giving or receiving award', 'golf chipping', 'golf driving', 'golf putting', 'grinding meat', 'grooming dog', 'grooming horse', 'gymnastics tumbling', 'hammer throw', 'headbanging', 'headbutting', 'high jump', 'high kick', 'hitting baseball', 'hockey stop', 'holding snake', 'hopscotch', 'hoverboarding', 'hugging', 'hula hooping', 'hurdling', 'hurling (sport)', 'ice climbing', 'ice fishing', 'ice skating', 'ironing', 'javelin throw', 'jetskiing', 'jogging', 'juggling balls', 'juggling fire', 'juggling soccer ball', 'jumping into pool', 'jumpstyle dancing', 'kicking field goal', 'kicking soccer ball', 'kissing', 'kitesurfing', 'knitting', 'krumping', 'laughing', 'laying bricks', 'long jump', 'lunge', 'making a cake', 'making a sandwich', 'making bed', 'making jewelry', 'making pizza', 'making snowman', 'making sushi', 'making tea', 'marching', 'massaging back', 'massaging feet', 'massaging legs', "massaging person's head", 'milking cow', 'mopping floor', 'motorcycling', 'moving furniture', 'mowing lawn', 'news anchoring', 'opening bottle', 'opening present', 'paragliding', 'parasailing', 'parkour', 'passing American football (in game)', 'passing American football (not in game)', 'peeling apples', 'peeling potatoes', 'petting animal (not cat)', 'petting cat', 'picking fruit', 'planting trees', 'plastering', 'playing accordion', 'playing badminton', 'playing bagpipes', 'playing basketball', 'playing bass guitar', 'playing cards', 'playing cello', 'playing chess', 'playing clarinet', 'playing controller', 'playing cricket', 'playing cymbals', 'playing didgeridoo', 'playing drums', 'playing flute', 'playing guitar', 'playing harmonica', 'playing harp', 'playing ice hockey', 'playing keyboard', 'playing kickball', 'playing monopoly', 'playing organ', 'playing paintball', 'playing piano', 'playing poker', 'playing recorder', 'playing saxophone', 'playing squash or racquetball', 'playing tennis', 'playing trombone', 'playing trumpet', 'playing ukulele', 'playing violin', 'playing volleyball', 'playing xylophone', 'pole vault', 'presenting weather forecast', 'pull ups', 'pumping fist', 'pumping gas', 'punching bag', 'punching person (boxing)', 'push up', 'pushing car', 'pushing cart', 'pushing wheelchair', 'reading book', 'reading newspaper', 'recording music', 'riding a bike', 'riding camel', 'riding elephant', 'riding mechanical bull', 'riding mountain bike', 'riding mule', 'riding or walking with horse', 'riding scooter', 'riding unicycle', 'ripping paper', 'robot dancing', 'rock climbing', 'rock scissors paper', 'roller skating', 'running on treadmill', 'sailing', 'salsa dancing', 'sanding floor', 'scrambling eggs', 'scuba diving', 'setting table', 'shaking hands', 'shaking head', 'sharpening knives', 'sharpening pencil', 'shaving head', 'shaving legs', 'shearing sheep', 'shining shoes', 'shooting basketball', 'shooting goal (soccer)', 'shot put', 'shoveling snow', 'shredding paper', 'shuffling cards', 'side kick', 'sign language interpreting', 'singing', 'situp', 'skateboarding', 'ski jumping', 'skiing (not slalom or crosscountry)', 'skiing crosscountry', 'skiing slalom', 'skipping rope', 'skydiving', 'slacklining', 'slapping', 'sled dog racing', 'smoking', 'smoking hookah', 'snatch weight lifting', 'sneezing', 'sniffing', 'snorkeling', 'snowboarding', 'snowkiting', 'snowmobiling', 'somersaulting', 'spinning poi', 'spray painting', 'spraying', 'springboard diving', 'squat', 'sticking tongue out', 'stomping grapes', 'stretching arm', 'stretching leg', 'strumming guitar', 'surfing crowd', 'surfing water', 'sweeping floor', 'swimming backstroke', 'swimming breast stroke', 'swimming butterfly stroke', 'swing dancing', 'swinging legs', 'swinging on something', 'sword fighting', 'tai chi', 'taking a shower', 'tango dancing', 'tap dancing', 'tapping guitar', 'tapping pen', 'tasting beer', 'tasting food', 'testifying', 'texting', 'throwing axe', 'throwing ball', 'throwing discus', 'tickling', 'tobogganing', 'tossing coin', 'tossing salad', 'training dog', 'trapezing', 'trimming or shaving beard', 'trimming trees', 'triple jump', 'tying bow tie', 'tying knot (not on a tie)', 'tying tie', 'unboxing', 'unloading truck', 'using computer', 'using remote controller (not gaming)', 'using segway', 'vault', 'waiting in line', 'walking the dog', 'washing dishes', 'washing feet', 'washing hair', 'washing hands', 'water skiing', 'water sliding', 'watering plants', 'waxing back', 'waxing chest', 'waxing eyebrows', 'waxing legs', 'weaving basket', 'welding', 'whistling', 'windsurfing', 'wrapping present', 'wrestling', 'writing', 'yawning', 'yoga', 'zumba', 'None']

if not os.path.exists('./image'):
    os.makedirs('./image')
else:
    shutil.rmtree('./image')
    os.makedirs('./image')    

state = 400 

def crop_image(re_img,new_height,new_width):
    re_img=Image.fromarray(np.uint8(re_img))
    width, height = re_img.size
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    crop_im = re_img.crop((left, top, right, bottom))
    crop_im = np.asarray(crop_im)
    return crop_im

def main():
    cmd = 'ffmpeg  -i \"{}\" -threads 1 -vf scale=-1:331 -q:v 0 \"{}/img_%05d.jpg\"'.format('./test.mp4', './image')
    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    files = os.listdir(r"./image/")
    files.sort(key=lambda x:int(x.split('img_')[1].split('.jpg')[0]))
    tick = len(files) / float(8)
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(8)])
    pil_img_list = list()
    for i in range(8):
        filename = files[int(offsets[i])]
        img = Image.open("./image/" + filename).convert('RGB')
        
        if img.width>img.height:
            frame_pil = img.resize((round(256*img.width/img.height),256))
        else:
            frame_pil = img.resize((256,round(256*img.height/img.width)))
        image = crop_image(frame_pil,224,224).transpose(2,0,1)
        imgs = [0,0,0]
        for i in range(3):
            imgs[0]=(image[0]/255-0.485)/0.229
            imgs[1]=(image[1]/255-0.456)/0.224
            imgs[2]=(image[2]/255-0.406)/0.225
        pil_img_list.extend([imgs])
    this_rst_list = []
    md = sdk.model(file_path, device_id)
    input = np.array(pil_img_list).astype(np.float32)
    t = sdk.Tensor(input)
    t.to_device(0)
    start_time = time.time()
    out = md.infer(t)
    cnt_time = time.time() - start_time
    out[0].to_host()
    out = out[0]
    out = np.array(out)
    this_rst_list.append(out)
    output_index = int(np.argmax(out))
    state = output_index
    print(cls_text[state])
    print('average {:.3f} sec/video'.format(float(cnt_time)))

if __name__ == '__main__':
    main()
