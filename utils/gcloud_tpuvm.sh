#!/bin/bash

gcloud alpha compute tpus tpu-vm create tpuvmv6e \
  --zone=europe-west4-a \
  --accelerator-type=v6e-8 \
  --version=v2-alpha-tpuv6e \
  --project=rag-finetunning \
  --spot 
  # --metadata-from-file=startup-script=cloud_clone.sh

sleep 30

gcloud alpha compute tpus tpu-vm ssh tpuvmv6e --zone=europe-west4-a


read -p "Do you want to delete the TPU VM? (y/n): " answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    echo "Deleting TPU VM..."
    gcloud alpha compute tpus tpu-vm delete tpuvmv6e \
      --zone=europe-west4-a \
      --project=rag-finetunning \
      --quiet
    echo "TPU VM deleted."
else
    echo "TPU VM kept running."
fi