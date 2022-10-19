FROM debian:sid
RUN apt-get update && apt-get install -y build-essential python3-pip curl && apt-get clean -y
#RUN apk add gcc musl-dev python3-dev
#RUN pip3 install --upgrade pip
RUN pip3 install matplotlib scikit-learn hdbscan numpy scipy pandas PrettyTable sympy

RUN mkdir Incremental_packing_clustering

ADD ./Incremental_packing_clustering_SERVER /Incremental_packing_clustering

RUN chmod 777 -R /Incremental_packing_clustering/logs
RUN chmod 777 -R /Incremental_packing_clustering/results

WORKDIR /Incremental_packing_clustering
