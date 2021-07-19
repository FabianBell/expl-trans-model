@echo off
set image_name=docker.siemens.com/fabian.bell.ext/expltransapi/api:latest
for /f "tokens=1,2 delims=:{} " %%A in (config.json) do (
	If "%%~B"=="true" (
		echo GPU is not supported for Windows
		exit /b 1
	)
)
docker run -p 81:80 -v %cd%\config.json:/root/config.json --name expltransapi --restart unless-stopped -d %image_name%
