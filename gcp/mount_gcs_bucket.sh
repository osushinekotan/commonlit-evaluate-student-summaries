export $(grep -v '^#' .env | xargs)

MOUNT_DIR="./resources"

gcloud config set project $PROJECT_ID
gcloud auth login

# checke bucket
if ! gsutil ls gs://$BUCKET_NAME; then
  gcloud storage buckets create gs://$BUCKET_NAME --location=us-central1
fi

# install gcsfuse 
if ! command -v gcsfuse &> /dev/null; then
  export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
  echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  sudo apt-get update
  sudo apt-get install -y gcsfuse
fi

# mount bucket at resources
if ! mountpoint -q $MOUNT_DIR; then
  gcsfuse $BUCKET_NAME $MOUNT_DIR
  echo "Mounted $BUCKET_NAME at $MOUNT_DIR."
else
  echo "$MOUNT_DIR is already mounted."
fi