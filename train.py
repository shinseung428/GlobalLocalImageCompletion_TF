import tensorflow as tf
from config import *
from network import *


def train(args, sess, model):
    #optimizers
    d_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_D").minimize(model.d_loss, var_list=model.d_vars)
    c_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_C").minimize(model.recon_loss, var_list=model.c_vars)
    
    global_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_C").minimize(model.loss, var_list=model.c_vars + model.d_vars)

    epoch = 0
    step = 0
    global_step = 0

    #saver
    saver = tf.train.Saver()        
    if args.continue_training:
        last_ckpt = tf.train.latest_checkpoint(args.checkpoints_path)
        saver.restore(sess, last_ckpt)
        ckpt_name = str(last_ckpt)
        print "Loaded model file from " + ckpt_name
        epoch = int(ckpt_name.split('-')[-1])
        tf.local_variables_initializer().run()
    else:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #summary init
    all_summary = tf.summary.merge([model.d_loss_sum,
                                    model.g_loss_sum,
                                    model.input_img_sum, 
                                    model.real_img_sum,
                                    model.recon_img_sum,
                                    model.g_local_imgs_sum,
                                    model.r_local_imgs_sum])
    writer = tf.summary.FileWriter(args.graph_path, sess.graph)


    #training starts here

    #first train completion network
    while epoch < args.epochs:
        #Update Completion Network
        summary, g_loss, _ = sess.run([all_summary, model.recon_loss, c_optimizer])
        writer.add_summary(summary, global_step)

        #Update Discriminator Networks
        # summary, d_loss, _ = sess.run([all_summary, model.d_loss, d_optimizer])
        # writer.add_summary(summary, global_step)


        # #Update All Networks
        # summary, g_loss, _ = sess.run([all_summary, model.loss, global_optimizer])
        # writer.add_summary(summary, global_step)       


        print "Epoch [%d] Step [%d] G Loss: [%.4f] D Loss: [%.4f]" % (epoch, step, g_loss, d_loss)

        if step*args.batch_size >= model.data_count:
            saver.save(sess, args.checkpoints_path + "/model", global_step=epoch)

            res_img = sess.run(model.X_g, feed_dict={model.z:batch_z})

            img_tile(epoch, args, res_img)
            step = 0
            epoch += 1 

        step += 1
        global_step += 1



    coord.request_stop()
    coord.join(threads)
    sess.close()            
    print("Done.")


def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    
    with tf.Session(config=run_config) as sess:
        model = network(args)

        print 'Start Training...'
        train(args, sess, model)

main(args)

#Still Working....