docker run -p 80:80 -v $(pwd)/config.json:/root/config.json --gpus all --name expltransapi --restart unless-stopped -d docker.siemens.com/fabian.bell.ext/expltransapi/api:latest
