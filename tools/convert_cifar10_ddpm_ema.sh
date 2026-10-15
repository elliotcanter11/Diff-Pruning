mkdir -p pretrained
if [ ! -f pretrained/cifar10-ema-model-790000.ckpt ]; then
    if command -v wget >/dev/null 2>&1; then
        wget -c "https://heibox.uni-heidelberg.de/f/2e4f01e2d9ee49bab1d5/?dl=1" -O pretrained/cifar10-ema-model-790000.ckpt
    else
        curl -L "https://heibox.uni-heidelberg.de/f/2e4f01e2d9ee49bab1d5/?dl=1" -o pretrained/cifar10-ema-model-790000.ckpt
    fi
fi

if [ ! -d pretrained/ddpm_ema_cifar10/unet ]; then
    python tools/convert_ddpm_original_checkpoint_to_diffusers_cifar10.py \
        --checkpoint_path pretrained/cifar10-ema-model-790000.ckpt \
        --config_file tools/ddpm_cifar10_config.json \
        --dump_path pretrained/ddpm_ema_cifar10
fi
