import sys
import json
import tensorflow as tf
import tensorflow_hub as hub
from runway import RunwayModel
from utils import truncated_z_sample, sample, CATEGORIES


biggan = RunwayModel()


@biggan.setup
def setup(architecture='256x256'):
  module_path = 'https://tfhub.dev/deepmind/biggan-{}/2'.format(architecture.split('x')[0])

  module = hub.Module(module_path)
  inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
            for k, v in module.get_input_info_dict().iteritems()}
  output = module(inputs)

  input_z = inputs['z']
  input_y = inputs['y']
  input_trunc = inputs['truncation']

  dim_z = input_z.shape.as_list()[1]
  vocab_size = input_y.shape.as_list()[1]

  initializer = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(initializer)

  return sess, vocab_size, dim_z, input_z, input_y, input_trunc, output


@biggan.command('sample', inputs={'truncation': 'float', 'category': 'text', 'seed': 'integer'}, outputs={'generatedOutput': 'image', 'z': 'vector'})
def sample_cmd(model, inp):
  (sess, vocab_size, dim_z, input_z, input_y, input_trunc, output) = model
  z = truncated_z_sample(1, dim_z, inp['truncation'], seed=inp['seed'])
  y = CATEGORIES.index(inp['category'])
  ims = sample(sess, z, y, vocab_size, input_z, input_y, input_trunc, output)
  return dict(generatedOutput=ims[0], z=z[0])


@biggan.command('generateFromVector', inputs={'z': 'vector', 'category': 'text'}, outputs={'generatedOutput': 'image'})
def generate_from_vector(model, inp):
  (sess, vocab_size, dim_z, input_z, input_y, input_trunc, output) = model
  z = inp['z'].reshape(-1, inp['z'].shape[0])
  y = CATEGORIES.index(inp['category'])
  ims = sample(sess, z, y, vocab_size, input_z, input_y, input_trunc, output)
  return dict(generatedOutput=ims[0])


if __name__ == '__main__':
    biggan.run()