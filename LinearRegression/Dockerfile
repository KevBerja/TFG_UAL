#Imagen oficial de python como principal
FROM python:3.8

#Instruccion que se ejecuta en el terminal del docker durante el proceso de creacion. Actualiza los paquetes
RUN apt-get update

#Define el directorio de trabajo /LinearRegression en el contenedor
WORKDIR /LinearRegression

#Copia el contenido de directorio actual al directorio /LinearRegression del contenedor
COPY . /LinearRegression

#Actualiza la herramienta pip instaladora de paquetes de python
RUN pip3 install --upgrade pip

#Instala los paquetes y dependencias python definidas en el fichero requeriments.txt
RUN pip3 install -r requirements.txt

#Se ejecuta al inicio del arranque del contenedor. El contenedor ejecutara comandos con la instruccion 'python' 
ENTRYPOINT ["python"]

#Servicio, archivo o script que lanzara el ENTRYPOINT al iniciarse el contenedor
CMD ["api.py"]                                                                                                                       