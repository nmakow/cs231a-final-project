from src.solver import CaptioningSolver
from src.model import CaptionGenerator
from src.utils import load_coco_data

def main():
    # load train dataset
    data = load_coco_data(split='train')

    print data["train_captions"][:10]

    model = CaptionGenerator(data["word_to_idx"],
                             dim_feature=[128, 512],
                             dim_embed=512,
                             dim_hidden=1024,
                             n_time_step=16,
                             prev2out=True,
                             ctx2out=True,
                             alpha_c=1.0,
                             selector=True,
                             dropout=True)

    solver = CaptioningSolver(model, data,
                              n_epochs=20,
                              batch_size=128,
                              update_rule="adam",
                              learning_rate=0.001,
                              print_every=100,
                              save_every=1,
                              pretrained_model=None,
                              model_path="model/lstm/",
                              test_model="model/lstm/model-10",
                              print_bleu=False,
                              log_path="./log/")

    solver.train()

if __name__ == "__main__":
    main()
