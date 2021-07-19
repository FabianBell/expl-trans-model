gpu=$(cat config.json | grep use_gpu | cut -d ':' -f 2 | tr -d ' ')
gpu_config=$(if [ $gpu == true ]; then echo '--gpus all'; fi)
docker run -p 81:80 -v $(pwd)/config.json:/root/config.json $gpu_config --name expltransapi --restart unless-stopped -d docker.siemens.com/fabian.bell.ext/expltransapi/api:latest
