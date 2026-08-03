python ddpm_sample.py \
--output_dir run/sample/ddpm_cifar10_pruned \
--batch_size ${2:-128} \
--total_samples ${1:-50000} \
--pruned_model_ckpt run/finetuned/ddpm_cifar10_pruned_post_training/pruned/unet_ema_pruned.pth \
--model_path run/finetuned/ddpm_cifar10_pruned_post_training \
--skip_type uniform \