FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN mkdir app
COPY requirements.txt setup.py example.py app/
COPY tom_rapperson app/tom_rapperson
RUN pip install -r app/requirements.txt
RUN pip install -U -e app

RUN pip install gdown
RUN gdown https://drive.google.com/uc?id=1NJQZK3kVZQdaqj2oVKkMmBoqfvIcCt24
RUN tar zxvf tom_rapperson.tar.gz model && rm tom_rapperson.tar.gz
RUN mv model app

RUN git clone https://github.com/nsu-ai/russian_g2p.git
RUN cd russian_g2p && pip install -U -e .

WORKDIR app
CMD ["/bin/bash"]
