#coding:utf-8
import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2, glob
import argparse
from networks.factory import get_network
import threadpool
import Queue
import threading
import time
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


queue=Queue.Queue()
num_thread=0



# 包括person类
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
FRAME_PATH='/share/dataset/UCF101-frames-TSN'
JSON_NAME='ucf_bboxes_of_person_dict.json'

def save_person_area(im,cls,dets,ax,image_name,video_name,thresh=0.5):
    '''
    # 功能是将image中所有person的bbox都存下来，
    # 如果帧中无人，则保存空列表
    '''
    bboxes=[] 
    for i in range(len(dets)):
        bbox=dets[i,:4].tolist()
        bboxes.append(bbox)
    return bboxes

def demo(sess, net, image_name,video_name):
    # ！！！！！！！！！！！！
    """Detect object classes in an image using pre-computed object proposals."""
    # 读入图片 
    im_file = os.path.join(FRAME_PATH, video_name, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()

    # from fast_rcnn.test import im_detect
    # scores 尺寸为(R,K),R为proposal的数量，K为类别数，每一行的K个数表示K个类别的得分
    # boxes 尺寸为(R,4*k)！
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    #print ('Detection took {:.3f}s for '
    #        '{:d} object proposals').format(timer.total_time, boxes.shape[0])  # 例：Detection took 0.1s for 10 object proposals'

    # Visualize detections for each class
    im2=im
    im = im[:, :, (2, 1, 0)] # 最后一个维度按从大到小排序存放，也就是每一个像素点的rgb的顺序发生改变
    ax=0

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    # 一类一类的获取检测结果
    for cls_ind, cls in enumerate(CLASSES[1:]): #对类中的每一项，获得索引和其类名
        if not cls=='person':
            continue
        cls_ind += 1 # because we skipped background 
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)] #存放第cls_ind类的4个数，表示位置信息（该类每个proposal的）
        cls_scores = scores[:, cls_ind] #存放第cls_ind类的得分（该类每个proposal的）
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32) #把信息合在一起
        keep = nms(dets, NMS_THRESH) #保留0.3的proposal！！
        dets = dets[keep, :]
 
        # 将对应图片的所有bbox存下来
        bboxes=save_person_area(im2,cls,dets,ax,image_name,video_name,thresh=0.8)
    return bboxes


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')
    
    args = parser.parse_args()

    return args


def vid_level_det(video_name):
    frame_level_bbox_dict={}
    #先获得视频路径
    video_path=os.path.join(FRAME_PATH,video_name)
    #再获得所有帧
    frames_path=os.path.join(FRAME_PATH,video_name,'img_*')
    frame_list=glob.glob(frames_path) #存放的是全路径
    
    # 逐帧检测
    for frame in frame_list:
        im_name=frame.split('/')[-1]
        #检测后保存至另一个文件夹中
        bboxes=demo(sess, net, im_name,video_name)
        frame_level_bbox_dict[im_name]=bboxes
    return frame_level_bbox_dict

def main():
    global sess
    global net
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # 调用本文件中parse_args函数来解析参数
    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))

    # load network 其实这一步也就是搭建计算图的过程 (from networks.factory import get_network)
    net = get_network(args.demo_net)  #args.demo_net默认是VGGnet_test
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

    # init session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3) #设置GPU占用率
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	sess.graph.finalize()
    	# load model
    	saver.restore(sess, args.model)
    	print '\n\nLoaded network {:s}'.format(args.model)

    	'''
    	im_names = ['test.jpg','000456.jpg', '000542.jpg', '001150.jpg',
                    '001763.jpg', '004545.jpg']
    	for im_name in im_names:
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'Demo for data/demo/{}'.format(im_name)
            # demo()是本文件内函数
            # 通过告知图片名称im_name(自能获得图片地址) sess以及net来使整个程序运行起来，并做后期处理
            demo(sess, net, im_name)
            break
    	#plt.show()
    	'''

    	# 读入数据. 首先获得所有视频的list（每一条都对应一个视频的名称）
        video_lists=os.listdir(FRAME_PATH) #listdir这个函数只列出名字，没有路径
        video_level_dict={}
        num=0
    	# 路径获得该视频下所有RGB图片的路径并读取，然后存到别的地方
    	for video_name in video_lists:
            print video_name
            '''
            #如果该视频已经检测完毕（即已经存在），则忽略（潜在bug，路径存在，但有可能帧数检测不全）
            if os.path.exists(os.path.join('/home/yzy/dataset/hmdb-det-part',video_name)):
                print('continue\n')
                continue
            '''            
            # 将该视频送入检测，frame_level_bbox_dict存放每一帧的bboxs,格式为{img1:[[bbox1], [bbox2], ...]}
            frame_level_bbox_dict=vid_level_det(video_name)
          
            # 记录该视频的检测结果，video_level_dict {vid1:{frame_level_bbox_dict}, vid2:{frame_level_bbox_dict}, ...}
            video_level_dict[video_name]=frame_level_bbox_dict
            
	    num+=1
            print('%dth video completed'%num)
            
            # 保存json
            if num%50==1:
                json.dump(video_level_dict,open(JSON_NAME,'w'))
            json.dump(video_level_dict,open(JSON_NAME,'w'))

            import gc;gc.collect()
	    from guppy import hpy;hxx = hpy();heap = hxx.heap()
            print(heap)
    	#pool=threadpool.ThreadPool(50)
    	#requests=threadpool.makeRequests(vid_level_det,video_lists)
    	#[pool.putRequest(req) for req in requests]
    	#pool.wait()
    	#print '%d second'% (time.time()-start_time)	




if __name__ == '__main__':
    main()
