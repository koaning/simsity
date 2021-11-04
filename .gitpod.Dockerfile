FROM python:3.7

COPY . .

RUN make install
