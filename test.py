import tensorflow as tf
from config import *
from network import *


drawing = False # true if mouse is pressed
ix,iy = -1,-1
color = (255,255,255)
size = 10

def erase_img(img_path):

    # mouse callback function
    def erase_rect(event,x,y,flags,param):
        global ix,iy,drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            if drawing == True:
                # cv2.circle(img,(x,y),10,(255,255,255),-1)
                cv2.rectangle(img,(x-size,y-size),(x+size,y+size),color,-1)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                # cv2.circle(img,(x,y),10,(255,255,255),-1)
                cv2.rectangle(img,(x-size,y-size),(x+size,y+size),color,-1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # cv2.circle(img,(x,y),10,(255,255,255),-1)
            cv2.rectangle(img,(x-size,y-size),(x+size,y+size),color,-1)


    img = cv2.imread(img_path)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',erase_rect)
    mask = np.zeros(img.shape)
    mask[img==255] = 1

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    print mask.shape
    cv2.imshow("mask",mask)
    cv2.waitKey()

    return img, mask


test_img = erase_img(args.img_path)

# def test(args, sess, model):
#     #saver
#     saver = tf.train.Saver()        
#     last_ckpt = tf.train.latest_checkpoint(args.checkpoints_path)
#     saver.restore(sess, last_ckpt)
#     ckpt_name = str(last_ckpt)
#     print "Loaded model file from " + ckpt_name
    
    

#     res_img = sess.run(model.test_res_imgs)

#     cv2.imshow("result", res_img)

#     print("Done.")


# def main(_):
#     run_config = tf.ConfigProto()
#     run_config.gpu_options.allow_growth = True
    
#     with tf.Session(config=run_config) as sess:
#         model = network(args)

#         print 'Start Testing...'
#         test(args, sess, model)

# main(args)

#Still Working....