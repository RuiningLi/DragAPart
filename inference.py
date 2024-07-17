import torch


def do_inference(
    model,
    diffusion,
    cond_latent: torch.FloatTensor,
    cond_clip: torch.FloatTensor,
    drags: torch.FloatTensor,
    cfg_scale: float = 1.0,
    latent_size: int = 32,
    latent_channel: int = 4,
):
    device = cond_latent.device
    num_samples = cond_latent.shape[0]
    z = torch.randn(
        num_samples, latent_channel, latent_size, latent_size, device=device
    )
    z = torch.cat([z, z], dim=0)
    cond_latent = torch.cat([cond_latent, cond_latent], dim=0)
    cond_clip = torch.cat([cond_clip, cond_clip], dim=0)
    drags = torch.cat([drags, drags], dim=0)

    model_kwargs = dict(
        x_cond=cond_latent,
        cfg_scale=cfg_scale,
        hidden_cls=cond_clip,
        drags=drags,
    )

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )
    samples, _ = samples.chunk(2, dim=0)
    return samples


if __name__ == "__main__":
    pass
