embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0)
)
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[5000, 10000],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_iters=500,
    warmup_ratio=0.001,
)

max_iters = 15000
runner = dict(type='IterBasedRunner', max_iters=max_iters)
