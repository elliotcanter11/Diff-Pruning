python ddpm_sample.py \
--output_dir run/sample/ddpm_cifar10_pretrained \
--batch_size ${2:-128} \
--total_samples ${1:-50000} \
--model_path pretrained/ddpm_ema_cifar10 \