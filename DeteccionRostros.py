#---------- Importamos las librerias ----------
import cv2
import mediapipe as mp

# Valida si la cámara se abrió con éxito
def validar_capturador(capturador: any):
    if (capturador.isOpened() == False):
        print("Error al abrir la secuencia de video o el archivo")

def cerrar():
    capturador_video.release()
    cv2.destroyAllWindows()

def mostrarPuntoCentral():
    for id, coordenadasXY in enumerate(result_rostros.detections):
        # Conversion de decimal a coordenada
        altoFrame, anchoFrame, c = frame.shape

        # Extraer x & y inicial
        x = coordenadasXY.location_data.relative_bounding_box.xmin
        y = coordenadasXY.location_data.relative_bounding_box.ymin

        # Extraer ancho y alto
        ancho = coordenadasXY.location_data.relative_bounding_box.width
        alto = coordenadasXY.location_data.relative_bounding_box.height

        # Conversion a pixeles
        xInicial, yInicial = int(x * anchoFrame), int(y * altoFrame)
        xFinal, yFinal = int(ancho * anchoFrame), int(alto * altoFrame)

        # Extraer el punto central de nuestro rostro
        xCentral = (xInicial + (xInicial + xFinal)) // 2
        yCentral = (yInicial + (yInicial + yFinal)) // 2

        # Mostrar coordenadas
        tamPunto = 8;
        cv2.circle(frame, (xCentral, yCentral), tamPunto, (255, 0, 255), cv2.FILLED)

        # Mostramos las coordenadas
        print("Coordenadas: X:", xInicial, " Y:", yFinal)

#---------- Declaraciones ----------
TECLA_SALIR = 27 #Tecla esc
TIEMPO_LECTURA_TECLA = 1
NIVEL_DETECCION = 0.75
color_puntos = (0, 255, 0)

detector = mp.solutions.face_detection #Detector
dibujo = mp.solutions.drawing_utils #Dibujo

capturador_video = cv2.VideoCapture(0) #1 es la webcam usb
validar_capturador(capturador_video)

#---------- Inicializamos parametros ----------
with detector.FaceDetection(min_detection_confidence = NIVEL_DETECCION) as rostros:
    while True:
        #Realizamos la lectura de la VideoCaptura
        isExito, frame = capturador_video.read()

        #Agregar efecto espejo
        frame = cv2.flip(frame, 1)

        #Correccion de color BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Almacena los rostros detectados
        result_rostros = rostros.process(frame_rgb)

        #Si hay rostros detectados
        if(result_rostros.detections is not None):
            #Recorre los rostros detectados
            for rostro in result_rostros.detections:
                dibujo.draw_detection(frame, rostro, dibujo.DrawingSpec(color=color_puntos))

                mostrarPuntoCentral()

        if(isExito):
            # Mostramos los fotogramas
            cv2.imshow("Webcam Reconocimiento facial", frame)

        #Leemos el teclado
        if cv2.waitKey(TIEMPO_LECTURA_TECLA) == TECLA_SALIR:  # Cuando es letra usar cv2.waitKey(25) & 0xFF == ord('q'):, para usar mayusucula o minuscula juntos
            break

cerrar()