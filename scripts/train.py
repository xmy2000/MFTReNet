import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging

from models.model_CGC_v2_multi import CGC, FRModel
from dataset.dataloader2 import get_dataloader

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    seed_everything(3407, workers=True)  # 3407

    # 数据准备
    train_path = "../data/dataset/train.pt"
    val_path = "../data/dataset/val.pt"
    test_path = "../data/dataset/test.pt"

    batch_size = 64
    train_loader = get_dataloader(train_path, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = get_dataloader(val_path, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = get_dataloader(test_path, batch_size=batch_size, shuffle=False, num_workers=8)
    print("=======================Prepare Data Finish!=======================")

    # 模型准备
    mtl_cgc = CGC(
        num_specific_experts=2,
        num_shared_experts=1,
        node_attr_dim=14,
        node_attr_emb=32,
        node_grid_dim=7,
        node_grid_emb=32,
        edge_attr_dim=15,
        edge_attr_emb=32,
        edge_grid_dim=12,
        edge_grid_emb=32,
        graph_encoder_layers=3
    )
    model = FRModel(
        mtl_cgc=mtl_cgc,
        classify_hidden_dim=64,
        segment_hidden_dim=16,
        rel_hidden_dim=16,
        num_classes=27,
        num_relations=8
    )

    print("=======================Model Summary=======================")
    summary = ModelSummary(model, max_depth=-1)
    print(summary)

    print("=======================Begin Train=======================")
    # 训练
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        filename='{epoch}-{cls_val_accuracy:.4f}-{seg_val_ap:.4f}-{rel_val_ap:.4f}'
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        check_on_train_epoch_end=False
    )
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-4, swa_epoch_start=20, annealing_epochs=10)

    trainer = L.Trainer(
        deterministic=True,
        max_epochs=100,
        default_root_dir="../checkpoints3/FRModel-final/",
        # profiler="simple",
        log_every_n_steps=10,
        precision="16-mixed",
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            swa_callback
        ]
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("=======================Begin Test=======================")
    # 测试
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")
