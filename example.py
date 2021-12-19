from tom_rhymer.rhymer import Rhymer

from tom_rapperson.encoder import SongsEncoder
from tom_rapperson.generator import SongsGenerator
from tom_rapperson.model import get_model_from_pl_checkpoint

if __name__ == '__main__':
    rhymer = Rhymer.load()
    encoder = SongsEncoder.load('./model')
    model = get_model_from_pl_checkpoint('./model/last.ckpt').to('cuda')
    gen = SongsGenerator(model, encoder, rhymer)
    for _ in range(20):
        print(gen.generate_verse() + '\n')
