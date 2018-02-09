import tensorflow as tf
from config import *
from ambientGAN import *

def test(args, sess, model):
    #saver
    saver = tf.train.Saver()        
    last_ckpt = tf.train.latest_checkpoint(args.checkpoints_path)
    saver.restore(sess, last_ckpt)
    ckpt_name = str(last_ckpt)
    print "Loaded model file from " + ckpt_name

    batch_z = np.random.uniform(-1, 1, size=(args.batch_size , args.input_dim))
    res_img = sess.run(model.X_g, feed_dict={model.z:batch_z})

    img_tile(999, args, res_img)

    print("Done.")


def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    
    with tf.Session(config=run_config) as sess:
        model = ambientGAN(args)
        args.images_path = os.path.join(args.images_path, args.measurement)
        args.graph_path = os.path.join(args.graph_path, args.measurement)
        args.checkpoints_path = os.path.join(args.checkpoints_path, args.measurement)

        #create graph and checkpoints folder if they don't exist
        if not os.path.exists(args.checkpoints_path):
            os.makedirs(args.checkpoints_path)
        if not os.path.exists(args.graph_path):
            os.makedirs(args.graph_path)
        if not os.path.exists(args.images_path):
            os.makedirs(args.images_path)

        print 'Start Testing...'
        test(args, sess, model)

main(args)

#Still Working....