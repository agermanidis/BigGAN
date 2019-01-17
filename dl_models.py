import tensorflow_hub as hub

module_path = 'https://tfhub.dev/deepmind/biggan-128/2'
module = hub.Module(module_path)

module_path = 'https://tfhub.dev/deepmind/biggan-256/2'
module = hub.Module(module_path)

module_path = 'https://tfhub.dev/deepmind/biggan-512/2'
module = hub.Module(module_path)