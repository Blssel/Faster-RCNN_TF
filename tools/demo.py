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


#CLASSES = ('__background__','person','bike','motorbike','car','bus')

def vis_detections(im, class_name, dets,ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]  #得分大于门限值的类保留其第一个维度的坐标
    print inds
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        print(bbox)
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def get_person_area(im,cls,dets,ax,image_name,video_name,thresh=0.5):
    '''
    # 功能是返回person的区域，
    # 如果帧中无人，则略过此帧
    # 如果帧中有超过一个人，则仅保留一个！！？？？？
    '''
    # 否则就把保留得分最高那个proposal的！！！！！！？？？？？？
    #inds=np.where(dets[:,-1]>=thresh)[0]
                        
    # 如果inds为空，则返回
    #if len(inds)==0:
    #    return
    # 不为空则取第一个,并利用其坐标信息得到ROI
    #else:
    bbox=dets[0,:4]
    return bbox.tolist()
'''
        #print bbox
        #获取roi
        roi=im[int(bbox[1]):int(bbox[3]),
              int(bbox[0]):int(bbox[2]),
              :]
        #r=im[100:150,0:10,:]
        #cv2.imshow(im)
        #roi=cv2.resize(roi,(224,224))
        #print('--------------------------------------------------------------------------------------------------------')
        if os.path.exists(os.path.join('/home/yzy/dataset/ucf-det',video_name))==False:
            os.mkdir(os.path.join('/home/yzy/dataset/ucf-det',video_name))
        save_path=os.path.join('/home/yzy/dataset/ucf-det',video_name,image_name)
        cv2.imwrite(save_path,roi)
        #cv2.imshow('test',roi)
        #cv2.waitKey(0)
        #print('hahahahahahhaahhaahahahha')
'''


def demo(sess, net, image_name,video_name):
    # ！！！！！！！！！！！！
    """Detect object classes in an image using pre-computed object proposals."""
    
    # 加载模型
    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im_file = os.path.join('/share/dataset/HMDB51-frames-TSN', video_name, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    # from fast_rcnn.test import im_detect
    # scores 尺寸为(R,K),R为proposal的数量，K为类别数
    # boxes 尺寸为(R,4*k)！！！！！！！！！！！！！！！
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    #print ('Detection took {:.3f}s for '
    #        '{:d} object proposals').format(timer.total_time, boxes.shape[0])  # 例：Detection took 0.1s for 10 object proposals'

    # Visualize detections for each class
    im2=im
    im = im[:, :, (2, 1, 0)] # 最后一个维度按从大到小排序存放，也就是每一个像素点的rgb的顺序发生改变
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    ax=0

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    for cls_ind, cls in enumerate(CLASSES[1:]): #对类中的每一项，获得索引和其类名
        cls_ind += 1 # because we skipped background 
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)] #存放第cls_ind类的4个数，表示位置信息（该类每个proposal的）
        cls_scores = scores[:, cls_ind] #存放第cls_ind类的得分（该类每个proposal的）
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32) #把信息合在一起
        keep = nms(dets, NMS_THRESH) #保留0.3的proposal！！
        dets = dets[keep, :]
        #vis_detections(im, cls, dets, ax, thresh=CONF_THRESH) # CONF_THRESH应该对保留的proposal的得分要求
        if cls!='person':
            continue
        bbox=get_person_area(im2,cls,dets,ax,image_name,video_name,thresh=0.8)
    return bbox


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
    #global queue
    #global num_thread
    #num_thread+=1
    #video_name=queue.get()
    FRAME_PATH='/share/dataset/HMDB51-frames-TSN'
    #FRAME_PATH='/share/dataset/UCF101-frames-TSN'
    #先获得视频路径
    video_path=os.path.join(FRAME_PATH,video_name)
    #再获得所有帧
    frames_path=os.path.join(FRAME_PATH,video_name,'img_*')
    frame_list=glob.glob(frames_path) #存放的是全路径
    #print(frame_list)
    for frame in frame_list:
        im_name=frame.split('/')[-1]
        #检测后保存至另一个文件夹中
        bbox=demo(sess, net, im_name,video_name)
        frame_level_bbox_dict[im_name]=bbox
        #print bbox
    return frame_level_bbox_dict
       
    #num_thread-=1

def main():
    global sess
    global net
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # 调用本文件中parse_args函数来解析参数
    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))

    # init session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
    net = get_network(args.demo_net)  #args.demo_net默认是VGGnet_test
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	sess.graph.finalize()
    	# load network  (from networks.factory import get_network)
    	#其实这一步也就是搭建计算图的过程
    	# load model
    	saver.restore(sess, args.model)

    	#sess.run(tf.initialize_all_variables())

    	print '\n\nLoaded network {:s}'.format(args.model)

    	# Warmup on a dummy image
    	#im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    	#for i in xrange(2):
        #    _, _= im_detect(sess, net, im)

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
    	# 读入数据

    	# 首先获得所有视频的list（每一条都对应一个视频的名称）
    	#FRAME_PATH='/share/dataset/UCF101-frames-TSN'
    	FRAME_PATH='/share/dataset/HMDB51_frames_TSN'
    	video_lists=os.listdir(FRAME_PATH)
    	# 路径获得该视频下所有RGB图片的路径并读取，然后存到别的地方
        video_level_dict={}
        num=0
    	for video_name in video_lists:
            print video_name
            #if os.path.exists(os.path.join('/home/yzy/dataset/ucf-det',video_name)):
            #    print('continue\n')
            #    continue
            frame_level_bbox_dict=vid_level_det(video_name)
            video_level_dict[video_name]=frame_level_bbox_dict
            num+=1
            print num
            if num%50==1:
                json.dump(video_level_dict,open('hmdb_bbox_of_person_dict.json','w'))
        json.dump(video_level_dict,open('hmdb_bbox_of_person_dict.json','w'))
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
