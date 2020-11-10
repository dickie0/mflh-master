from .mflh import MFLH
from .util import Dataset


def train(tr_imgs, db_imgs, q_imgs, config):
    model = MFLH(config)
    img_database = Dataset(db_imgs, config.output_dim)
    img_query = Dataset(q_imgs, config.output_dim)
    img_train = Dataset(tr_imgs, config.output_dim)
    model.training(img_train, img_query, img_database, config.number)


def test(db_imgs, q_imgs, config):
    model = MFLH(config)
    img_database = Dataset(db_imgs, config.output_dim)
    img_query = Dataset(q_imgs, config.output_dim)
    return model.test(img_query, img_database, config.number)
