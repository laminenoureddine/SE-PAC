FROM debian:sid
RUN apt-get update && apt-get install -y build-essential python3-pip curl && apt-get clean -y
#RUN apk add gcc musl-dev python3-dev
#RUN pip3 install --upgrade pip
RUN pip3 install matplotlib scikit-learn==0.20 numpy scipy pandas PrettyTable sympy

RUN mkdir mds
RUN mkdir mds/mds_results

COPY * /mds/

RUN chmod 777 -R /mds/mds_results

WORKDIR /mds
