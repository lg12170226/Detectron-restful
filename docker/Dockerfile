# Use Caffe2 image as parent image
FROM caffe2:cuda8-cudnn6-all-options

# Install Python dependencies
RUN pip install numpy pyyaml matplotlib opencv-python>=3.0 setuptools Cython mock

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR /cocoapi/PythonAPI
RUN make install

# Clone the Detectron repository
#RUN git clone https://github.com/facebookresearch/detectron /detectron
RUN git clone https://github.com/lg12170226/Detectron-restful /detectron
# Set up Python modules
WORKDIR /detectron/lib
RUN make

# Build custom ops
RUN make ops

# Go to Detectron root
WORKDIR /detectron
RUN apt-get install python-pip
RUN pip install --upgrade pip
RUN pip install Flask


WORKDIR /detectron/restful
CMD python app.py

