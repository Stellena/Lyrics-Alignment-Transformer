class CFG:
    def __init__(self):
        self.PAD = 0
        self.EOS = 1
        self.MAXLEN = 2048
        self.BATCH_SIZE = 8          # 너무 크게 잡으면 램용량 초과 뜰 수 있음.
        self.NUM_TOKENS = 755        # 토큰 종류 수
        self.dim_model = 64
        self.num_heads = 4
        self.num_layers = 4
        self.log_period = 500
        self.input_train_dir = "dataset/input_train_lyrics.npy"
        self.target_train_dir = "dataset/target_train_lyrics.npy"
        self.input_val_dir = "dataset/input_val_lyrics.npy"
        self.target_val_dir = "dataset/target_val_lyrics.npy"
        self.save_checkpoint_dir = "checkpoint.pt"
    