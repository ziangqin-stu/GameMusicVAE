import tensorflow as tf
import numpy as np
import trained_model
import os
import midi2notesequence
import modelpy


GAME_FOLDERS = '/data'
game_dirs = os.listdir(GAME_FOLDERS)

class MIvae():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.trainedmodel = TrainedModel('hierdec-mel_16bar',self.batch_size, 'check_dir')
        self.img_list, self.midi_list = self.load_data()
        sefl.cnn = cnn()

    def cnn(self, input):



    def load_data(self):
        img_list = []
        midi_list = []
        for i in range(len(game_dirs)):
            game_path = os.path.join(GAME_FOLDERS,game_dirs[i])
            files = os.listdir(game_path).sort()
            file_num = len(files)
            for j in range(0, file_num, 2):
                img_name = files[j+1]
                img = np.load(os.path.join(game_path,img_name))
                img_num = img.shape[0]
                midi_name = files[j]
                img_data = img.reshape((-1,256,256,3))


                midi_cut = cut_midi(midi_name, img_num)  #需要返回list of notesquence object 9N个

                for k in range(img_name*9):
                    img_list.append(img_data[k,:,:,:])
                    midi_list.append(midi_cut[k])

        return img_list,midi_list


    def get_batches(self, i):
        img_list = self.img_list
        midi_list =self.midi_list
        batch_num = len(img_list)//self.batch_size
        return


    def train(self):
        input = tf.placeholder('float', shape=[None, 256, 256, 3])
        label_z = tf.placeholder('float', shape=[None, 512])
        gen_z = self.cnn(input)
        loss = tf.reduce_mean(tf.norm(gen_z-label_z))
        train_op = tf.train.AdamOptimizer().minimize(loss)
        losses =[]
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for epoch in range(500):
                for i in range(len(self.img_list)//self.batch_size):
                    img_batch = self.img_list[i*self.batch_size:(i+1)*self.batch_size]
                    midi_batch = self.img_list[i*self.batch_size:(i+1)*self.batch_size]
                    z = self.trainedmodel.encode(midi_batch)[0] # label_z
                    print('epoch '+str(epoch)+' loss = '+str(loss))
                    session.run(train_op, feed_dict={input: img_batch, label_z: z})

    def generate(self, img):
        z = self.cnn(img)
        note_seq = self.trainedmodel.decode(z)
        midi_file = note_seq_tomidi(note_seq)
        return midifile







