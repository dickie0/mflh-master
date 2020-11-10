import time
from math import ceil
from datetime import datetime
from tools.ops import *
from tools.util import *
from utils.network import *
from tools.params import params_with_name
from utils.metrics.tf_v import metric
from utils.evaluation import MAP


class MFLH(object):
    def __init__(self, config):
        np.set_printoptions(precision=4)
        self.output_dim = config.output_dim
        self.n_class = config.label_dim
        self.gpu = config.gpus
        self.batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.max_epoch = config.epochs
        self.dist = config.dist
        self.lr = config.lr
        self.alpha = config.alpha
        self.eta = config.eta
        self.gamma = config.gamma
        self.scale = config.scale_factor
        self.triplet_margin = config.triplet_margin
        self.strategy = config.strategy
        self.dataset = config.dataset
        self.data_dir = config.data_dir
        self.model_weights = config.model_weights
        self.decay_steps = config.decay_step
        self.decay_rate = config.decay_rate
        self.beta = tf.Variable(0, dtype=tf.float32)
        self.global_ = tf.placeholder(dtype=tf.int32)
        self.bb = tf.add(self.beta, 1.)
        self.update = tf.assign(self.beta, self.bb)
        self.Num = config.number
        self.height = self.width = config.size
        self.model_path = config.pre_model_path

        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        config_proto.allow_soft_placement = True
        self.sess = tf.Session(config=config_proto)
        self.img = tf.placeholder(tf.float32, [None, 3072])
        self.label = tf.placeholder(tf.float32, shape=[3*self.batch_size, self.n_class])
        self.mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img-mean')
        self.ori_img = self.data_processing(self.img, self.height, self.width, self.mean)
        self.trans_img1, self.trans_img2 = self.csm_operation(self.ori_img, self.batch_size,
                                                              self.height, self.width, scale=self.scale)

        self.f_g, self.logic_g \
            = encoder(self.ori_img, self.batch_size, self.model_weights,
                      lamb=self.beta, output_dim=self.output_dim, branch='g', stage='train')
        self.f_h, self.logic_h \
            = encoder(self.trans_img1, self.batch_size, self.model_weights,
                      lamb=self.beta, output_dim=self.output_dim, branch='h', stage='train')
        self.f_v, self.logic_v \
            = encoder(self.trans_img2, self.batch_size, self.model_weights,
                      lamb=self.beta, output_dim=self.output_dim, branch='v', stage='train')

        self.f_hr = tf.concat([i for i in self.f_h[1:]], axis=1)
        self.f_vr = tf.concat([i for i in self.f_v[1:]], axis=1)
        self.f_block = tf.concat([self.f_h[0], self.f_hr, self.f_vr, self.f_v[0]], axis=1)
        self.f_locals = linear('encoder.l', self.output_dim * 6, self.output_dim, self.f_block)
        self.f_m = tf.add(self.f_g, tf.multiply(self.alpha, self.f_locals))

        self.a_label, self.p_label, self.n_label = tf.split(self.label, 3, axis=0)
        self.cls_h = tf.reduce_sum([tf.losses.softmax_cross_entropy(
            self.label, i, weights=1.0) for i in self.logic_h])
        self.cls_v = tf.reduce_sum([tf.losses.softmax_cross_entropy(
            self.label, i, weights=1.0) for i in self.logic_v])
        self.cls_g = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logic_g, labels=self.label))
        self.cls_loss = self.cls_h + self.cls_v + self.cls_g
        self.trip_mul = self.triplet_loss(self.f_m, self.triplet_margin)
        self.trip_g = self.triplet_loss(self.f_g, self.triplet_margin)
        self.trip_l = self.triplet_loss(self.f_locals, self.triplet_margin)
        self.trip_loss = self.trip_mul + self.trip_g + self.trip_l
        self.pw_sim_loss = self.pair_similarity_loss(self.f_m, self.label, gam=self.gamma)
        self.total_loss = self.cls_loss + self.trip_loss + self.eta * self.pw_sim_loss

        self.encode_params = params_with_name('encoder')
        self.rr = tf.train.exponential_decay(self.lr, self.global_, self.decay_steps, self.decay_rate, staircase=True)
        self.train_op_encode = tf.train.AdamOptimizer(learning_rate=self.rr, beta1=0.9, beta2=0.999) \
            .minimize(self.total_loss, var_list=self.encode_params)

        self.img_nor = self.normalization(self.img, stage='val')
        self.te_im = self.val_data_processing(self.img_nor, self.height, self.width, self.mean)
        self.f_te, _ \
            = encoder(self.te_im, self.test_batch_size * 10, self.model_weights, self.output_dim, stage='val')
        self.sess.run(tf.global_variables_initializer())
        return

    @staticmethod
    def csm_op(img, h_width, alpha):
        fu = [tf.image.resize_images(img[alpha * i:], (h_width, h_width)) for i in
              range(1, round((h_width / alpha) / 2))]
        bd = [tf.image.resize_images(img[:alpha * i], (h_width, h_width)) for i in
              range(round((h_width / alpha) / 2), int(h_width / alpha))]
        fr = [tf.image.resize_images(img[:, :-alpha * i], (h_width, h_width)) for i in
              range(1, round((h_width / alpha) / 2))]
        bl = [tf.image.resize_images(img[:, -alpha * i:], (h_width, h_width)) for i in
              range(round((h_width / alpha) / 2), int(h_width / alpha))]
        if round((h_width / alpha) / 2) == 1:
            output = tf.concat([bd, bl], axis=0)
        else:
            output = [tf.concat([fu, bd, fr, bl], axis=0)]
        return output

    def normalization(self, x, stage="train"):
        x = 2 * tf.cast(x, tf.float32) / 256. - 1
        if stage == "train":
            x += tf.random_uniform(shape=[3*self.batch_size, 3072], minval=0., maxval=1. / 128)
        else:
            x += tf.random_uniform(shape=[self.test_batch_size, 3072], minval=0., maxval=1. / 128)
        return x

    def data_processing(self, image, height, width, m):
        img_norm = self.normalization(image, stage='train')
        im_resize = resize_img(img_norm, 32)
        im_trans = tf.cast(im_resize, tf.float32)[:, :, :, ::-1]
        im_d = tf.stack([tf.random_crop(tf.image.random_flip_left_right(each_image), [height, width, 3])
                         for each_image in tf.unstack(im_trans, 3*self.batch_size)])
        return im_d - m

    def csm_operation(self, image, batch_size, height, width, scale, order=True):
        ud_rl = [tf.random_shuffle(self.csm_op(im, int(image.shape[1]), height // (scale + 1))[0])
                 for im in tf.unstack(image, 3*batch_size)]
        if scale == height // 2:
            trans_1 = [tf.concat([ud_rl[i]], 0) for i in range(int(3*batch_size))]
            trans_2 = [tf.concat([ud_rl[i]], 0) for i in range(int(3*batch_size))]
        else:
            trans_1 = [tf.concat([ud_rl[i][0]], 0) for i in range(int(3*batch_size))]
            trans_2 = [tf.concat([ud_rl[i][1]], 0) for i in range(int(3*batch_size))]
        trans_1 = tf.reshape(trans_1, (3*batch_size, height, width, 3))
        trans_2 = tf.reshape(trans_2, (3*batch_size, height, width, 3))
        return trans_1, trans_2

    def pair_similarity_loss(self, embeddings, labels, gam=0.1, reg=False):
        u, v, w = tf.split(embeddings, 3)
        label_u, label_v, label_w = tf.split(labels, 3)
        label_aa = tf.cast(tf.matmul(label_u, tf.transpose(label_u)), tf.float32)
        label_ap = tf.cast(tf.matmul(label_u, tf.transpose(label_v)), tf.float32)
        label_pp = tf.cast(tf.matmul(label_v, tf.transpose(label_v)), tf.float32)
        s_aa = tf.clip_by_value(label_aa, 0.0, 1.0)
        s_pp = tf.clip_by_value(label_pp, 0.0, 1.0)
        s_ap = tf.clip_by_value(label_ap, 0.0, 1.0)
        aa = tf.matmul(u, tf.transpose(u))
        ap = tf.matmul(u, tf.transpose(v))
        pp = tf.matmul(v, tf.transpose(v))
        sim_l = tf.reduce_mean(tf.square(0.5 * (ap + self.output_dim) - self.output_dim * s_ap))
        sim_l += tf.reduce_mean(tf.square(0.5 * (aa + self.output_dim) - self.output_dim * s_aa))
        sim_l += tf.reduce_mean(tf.square(0.5 * (pp + self.output_dim) - self.output_dim * s_pp))
        if reg:
            regular = tf.reduce_mean(tf.abs(tf.abs(embeddings) - tf.constant(1.0)))
            pw_s = sim_l + gam * regular
        else:
            pw_s = sim_l
        return pw_s

    def triplet_loss(self, embeddings, margin):
        with tf.variable_scope('triplet_loss'):
            anchor, pos, neg = tf.split(embeddings, 3)
            pos_dist = metric(anchor, pos, pair=False, dist_type=self.dist)
            neg_dist = metric(anchor, neg, pair=False, dist_type=self.dist)
            basic_loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
            dist_loss = tf.reduce_mean(basic_loss, 0)
        return dist_loss

    def val_data_processing(self, im, height, width, m):
        im = resize_img(im, 32)
        im = tf.cast(im, tf.float32)[:, :, :, ::-1]
        image = tf.unstack(im, self.test_batch_size)

        def crop(img, x, y):
            return tf.image.crop_to_bounding_box(img, x, y, height, width)

        def distort(f, x, y):
            return tf.stack([crop(f(each), x, y) for each in image])

        def distort_raw(x, y):
            return distort(lambda x: x, x, y)

        def distort_flipped(x, y):
            return distort(tf.image.flip_left_right, x, y)

        val_distorted = tf.concat([distort_flipped(0, 0), distort_flipped(28, 0),
                                   distort_flipped(0, 28), distort_flipped(28, 28),
                                   distort_flipped(14, 14), distort_raw(0, 0),
                                   distort_raw(28, 0), distort_raw(0, 28),
                                   distort_raw(28, 28), distort_raw(14, 14)], 0)
        val_image = val_distorted - m
        return val_image

    def triplets_selecting(self, img_dataset):
        its = int(img_dataset.n_samples / (3*self.batch_size))
        for i in range(its):
            images, labels, _ = img_dataset.new_batch(3*self.batch_size)
            output = self.sess.run(self.f_g, feed_dict={self.img: images, self.label: labels})
            img_dataset.batch_out(3*self.batch_size, output)
        img_dataset.iter_triplets(n_part=20, dist='euclidean2', select_strategy=self.strategy)

    def training(self, img_dataset, img_query, img_database, number):
        # ---training---
        print("Start training: MFLH")
        start = time.time()
        self.triplets_selecting(img_dataset)
        train_iter = 0
        for epoch in range(self.max_epoch):
            self.sess.run(self.update)
            triplet_bs = int(self.batch_size)
            max_iter = int(img_dataset.triplets.shape[0] / triplet_bs)
            img_dataset.finish_epoch()
            for i in range(max_iter):
                images, labels = img_dataset.new_triplet(triplet_bs)
                _, output, t_loss = self.sess.run([self.train_op_encode, self.f_g, self.total_loss],
                                                  feed_dict={self.img: images, self.label: labels,
                                                             self.global_: epoch})
                img_dataset.batch_triplet(triplet_bs, output)
                if i % int(max_iter / 2) == 0 or (i + 1) % max_iter == 0:
                    print('Epoch: [%d/%d][%d/%d]\t\t Time: %.3fs\t\t loss: %.3f'
                          % (epoch, self.max_epoch, i + 1, max_iter, (time.time() - start) / 60., t_loss))
                train_iter += 1
            self.triplets_selecting(img_dataset)

            maps = self.test(img_query, img_database, number)
            for key in maps:
                print("{}\t{}".format(key, maps[key]))
        self.sess.close()

    def test(self, img_query, img_database, number):
        self.test_batch(img_query)
        self.test_batch(img_database)
        maps = MAP(number)
        return {'MAP:': maps.map_fr(img_database, img_query), 'MAP_s:': maps.map_sign(img_database, img_query)}

    def test_batch(self, img_dataset):
        batch = int(ceil(img_dataset.n_samples / float(self.test_batch_size)))
        img_dataset.finish_epoch()
        for i in range(batch):
            images, labels, _ = img_dataset.new_batch(self.test_batch_size)
            output = self.sess.run([self.f_te], feed_dict={self.img: images})
            img_dataset.batch_out(self.test_batch_size, output)
            if (i + 1) % int(batch / 2) == 0 or (i + 1) % batch == 0:
                print("%s batch [%d]/[%d]" % (datetime.now(), i, batch))
