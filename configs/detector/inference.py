default_scope = 'mmdet'
model = dict(
    as_two_stage=True,
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=128,
        frozen_stages=-1,
        init_cfg=None,
        mlp_ratio=4,
        num_heads=[
            4,
            8,
            16,
            32,
        ],
        out_indices=(
            1,
            2,
            3,
        ),
        patch_norm=True,
        pretrain_img_size=384,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=12,
        with_cp=True),
    bbox_head=dict(
        contrastive_cfg=dict(bias=True, log_scale='auto', max_text_len=256),
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=2,
        sync_cls_avg_factor=True,
        type='GroundingDINOHead_ParallelDecoder_DN'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=False,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            cross_attn_text_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        fusion_layer_cfg=dict(
            embed_dim=1024,
            init_values=0.0001,
            l_dim=256,
            num_heads=4,
            v_dim=256),
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_cp=6,
        num_layers=6,
        text_layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=4))),
    language_model=dict(
        add_pooling_layer=False,
        max_tokens=256,
        name='{{ fileDirname }}/../../assets/tokenizer/bert-base-uncased',
        weights_path='{{ fileDirname }}/../../weights/bert.safetensors',
        pad_to_max=False,
        special_tokens_list=[
            '[CLS]',
            '[SEP]',
            '.',
            '?',
        ],
        type='BertModel',
        use_sub_sentence_represent=True),
    neck=dict(
        act_cfg=None,
        bias=True,
        in_channels=[
            256,
            512,
            1024,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    rand_dnquery_rate=0.5,
    test_cfg=dict(max_per_img=100),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='GroundingDINO_ParallelDecoder_15_DNQuery_rand',
    use_autocast=True,
    with_box_refine=True)
