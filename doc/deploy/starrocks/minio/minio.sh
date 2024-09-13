docker run -d \
   --name minio \
   --user $(id -u):$(id -g) \
   -p 9001:9001 \
   -e "MINIO_ROOT_USER=ROOTUSER" \
   -e "MINIO_ROOT_PASSWORD=CHANGEME123" \
   -v ${HOME}/projects/minio/data:/data \
   minio/minio server /data --console-address ":9001"
