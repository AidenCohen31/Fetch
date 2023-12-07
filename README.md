RUN APP
docker build -t ml .
docker run -p 5000:5000 ml


ACCESS APP IN BROWSER AT localhost:5000
